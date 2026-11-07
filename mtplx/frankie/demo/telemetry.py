"""First-audio latency, by stage, sent to the client as ``frankie.latency``.

Each response gets wall-clock marks, in ms since ``Session.start`` queued it:
``executor`` (the owner thread picked it up), ``asr``/``asr_end`` (Parakeet on
this turn's input), ``owner`` (``Frankie.respond`` entry), ``closed`` (the
closed-history prefill finished), ``decode`` (the generation call, whose
header prefill precedes the first token), ``token``, ``tts``, ``audio`` (the
first generated PCM), ``release`` (the turn was committed) and ``published``
(the first PCM left for the client). ``stages`` are the gaps between them.
Timestamps are taken on the thread that does the work. Nothing here
synchronizes, samples or changes inference.
"""
import functools
import threading
import time
import weakref

STAGES = {
    "queue_ms": ("queued", "executor"),
    "asr_ms": ("asr", "asr_end"),
    "closed_prefill_ms": ("owner", "closed"),
    "header_first_token_ms": ("decode", "token"),
    "first_phrase_ms": ("token", "tts"),
    "tts_first_chunk_ms": ("tts", "audio"),
    "held_ms": ("audio", "published"),
    "release_to_audio_ms": ("release", "published"),
}


def install():
    from mtplx.frankie import engine
    from mtplx.frankie.session import Session

    owner = threading.local()
    inputs = weakref.WeakValueDictionary()  # input item id -> run.abort, until respond takes over
    original_start = Session.start
    original_show = Session.show
    original_publish = Session.publish
    original_respond = engine.Frankie.respond
    original_prepare = engine.Frankie.prepare_audio
    original_speak = engine.AudioModels.speak
    original_warm = engine.Frankie.warm
    sessions = weakref.WeakValueDictionary()  # session id -> Session, for idle warm reports

    def mark(record, name):
        if record is not None and name not in record:
            record[name] = time.monotonic()

    @functools.wraps(original_start)
    def start(self, *args, **kwargs):
        queued = time.monotonic()
        run = original_start(self, *args, **kwargs)
        # start() is synchronous: its executor task cannot run until it returns.
        run.abort._frankie_timing = {
            "queued": queued, "response_id": run.id,
            "input_item_id": (run.input or {}).get("id"), "session": weakref.ref(self),
        }
        sessions[self.id] = self
        if run.input is not None:
            inputs[run.input["id"]] = run.abort
        if run.visible:
            mark(run.abort._frankie_timing, "release")
        return run

    @functools.wraps(original_show)
    def show(self, run):
        original_show(self, run)
        if run.visible:
            mark(getattr(run.abort, "_frankie_timing", None), "release")

    @functools.wraps(original_publish)
    def publish(self, run, kind, value):
        original_publish(self, run, kind, value)
        record = getattr(run.abort, "_frankie_timing", None)
        if kind == "audio" and record is not None and "published" not in record and run.visible:
            mark(record, "published")
            _publish(record)

    @functools.wraps(original_prepare)
    def prepare_audio(self, item):
        record = getattr(owner, "record", None)
        if record is None:
            abort = inputs.get(item.get("id"))
            record = getattr(abort, "_frankie_timing", None)
            mark(record, "executor")
        fresh = any(p["type"] == "input_audio" and "_transcript" not in p for p in item.get("content", []))
        if fresh:
            mark(record, "asr")
        try:
            return original_prepare(self, item)
        finally:
            if fresh:
                mark(record, "asr_end")

    @functools.wraps(original_warm)
    def warm(self, settings, *, session_id, **kwargs):
        started, outcome, stats = time.monotonic(), "warmed", {}
        try:
            result = original_warm(self, settings, session_id=session_id, **kwargs)
            if result is None:
                outcome = "skipped"
            else:
                stats = {k: v for k, v in result.stats.to_dict().items()
                         if k in ("cached_tokens", "new_prefill_tokens", "prompt_eval_time_s")}
            return result
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            session = sessions.get(session_id)
            if session is not None:
                row = dict(outcome=outcome, ms=round((time.monotonic() - started) * 1000, 3),
                           history_items=len(kwargs.get("history", ())), **stats)
                session.loop.call_soon_threadsafe(
                    lambda: None if session.closed else session.event("frankie.history_warm", **row))

    @functools.wraps(original_respond)
    def respond(self, items, settings, emit, abort, **kwargs):
        previous = getattr(owner, "record", None)
        record = getattr(abort, "_frankie_timing", None)
        owner.record = record
        mark(record, "executor")
        mark(record, "owner")
        try:
            return original_respond(self, items, settings, emit, abort, **kwargs)
        finally:
            owner.record = previous

    def generation(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            record = getattr(owner, "record", None)
            callback = kwargs.get("token_callback")
            if record is None or "decode" in record:
                return original(*args, **kwargs)
            if callback is None:  # the closed-history prefill
                try:
                    return original(*args, **kwargs)
                finally:
                    mark(record, "closed")
            mark(record, "decode")

            def committed(*values, **options):
                if values and len(values[0]):
                    mark(record, "token")
                return callback(*values, **options)

            kwargs["token_callback"] = committed
            return original(*args, **kwargs)
        return wrapped

    @functools.wraps(original_speak)
    def speak(self, *args, **kwargs):
        record = getattr(owner, "record", None)
        first = record is not None and "tts" not in record
        mark(record if first else None, "tts")
        stream = original_speak(self, *args, **kwargs)
        try:
            for pcm in stream:
                if first and len(pcm):
                    first = False
                    mark(record, "audio")
                yield pcm
        finally:
            stream.close()

    Session.start = start
    Session.show = show
    Session.publish = publish
    engine.Frankie.respond = respond
    engine.Frankie.prepare_audio = prepare_audio
    engine.Frankie.warm = warm
    engine.AudioModels.speak = speak
    engine.generate_ar = generation(engine.generate_ar)
    engine.generate_mtpk = generation(engine.generate_mtpk)


def _publish(record):
    stages = {name: round((record[b] - record[a]) * 1000, 3)
              for name, (a, b) in STAGES.items() if a in record and b in record}
    marks = {name: round((value - record["queued"]) * 1000, 3) for name, value in record.items()
             if isinstance(value, float) and name != "queued" and "queued" in record}
    session = record["session"]()
    if session is None:
        return

    def publish():
        if not session.closed:
            session.event("frankie.latency", response_id=record["response_id"],
                          input_item_id=record["input_item_id"], stages=stages, marks=marks)
    session.loop.call_soon_threadsafe(publish)
