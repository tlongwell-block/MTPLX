"""First-audio latency, by stage, sent to the client as ``frankie.latency``.

For each response: queue wait, input prefill, first brain token, first phrase,
and Breeze's first chunk. Timestamps are taken on the thread that does the
work. Nothing here synchronizes, samples or changes inference.
"""
import functools
import threading
import time
import weakref

STAGES = {
    "queue_ms": ("queued", "owner"),
    "input_prefill_ms": ("owner", "decode"),
    "first_token_ms": ("decode", "token"),
    "first_phrase_ms": ("token", "tts"),
    "tts_first_chunk_ms": ("tts", "audio"),
}


def install():
    from mtplx.frankie import engine
    from mtplx.frankie.session import Session

    owner = threading.local()
    original_start = Session.start
    original_respond = engine.Frankie.respond
    original_speak = engine.AudioModels.speak

    @functools.wraps(original_start)
    def start(self, *args, **kwargs):
        queued = time.monotonic()
        run = original_start(self, *args, **kwargs)
        # start() is synchronous: its executor task cannot run until it returns.
        run.abort._frankie_timing = {
            "queued": queued, "response_id": run.id,
            "input_item_id": (run.input or {}).get("id"), "session": weakref.ref(self),
        }
        return run

    @functools.wraps(original_respond)
    def respond(self, items, settings, emit, abort, **kwargs):
        previous = getattr(owner, "record", None)
        record = getattr(abort, "_frankie_timing", None)
        owner.record = record
        if record is not None:
            record["owner"] = time.monotonic()
        try:
            return original_respond(self, items, settings, emit, abort, **kwargs)
        finally:
            owner.record = previous

    def generation(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            record = getattr(owner, "record", None)
            callback = kwargs.get("token_callback")
            # Ignore cache-only passes and unrelated background work.
            if record is not None and callback is not None and "decode" not in record:
                record["decode"] = time.monotonic()

                def committed(*values, **options):
                    if values and len(values[0]) and "token" not in record:
                        record["token"] = time.monotonic()
                    return callback(*values, **options)

                kwargs["token_callback"] = committed
            return original(*args, **kwargs)
        return wrapped

    @functools.wraps(original_speak)
    def speak(self, *args, **kwargs):
        record = getattr(owner, "record", None)
        first = record is not None and "tts" not in record
        if first:
            record["tts"] = time.monotonic()
        stream = original_speak(self, *args, **kwargs)
        try:
            for pcm in stream:
                if first and len(pcm):
                    first = False
                    record["audio"] = time.monotonic()
                    _publish(record)
                yield pcm
        finally:
            stream.close()

    Session.start = start
    engine.Frankie.respond = respond
    engine.AudioModels.speak = speak
    engine.generate_ar = generation(engine.generate_ar)
    engine.generate_mtpk = generation(engine.generate_mtpk)


def _publish(record):
    stages = {name: round((record[b] - record[a]) * 1000, 3)
              for name, (a, b) in STAGES.items() if a in record and b in record}
    session = record["session"]()
    if session is None:
        return

    def publish():
        if not session.closed:
            session.event("frankie.latency", response_id=record["response_id"],
                          input_item_id=record["input_item_id"], stages=stages)
    session.loop.call_soon_threadsafe(publish)
