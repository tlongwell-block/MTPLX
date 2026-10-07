"""AudioSeal watermark on Frankie's final speech PCM.

``install`` wraps ``Frankie.respond`` so every audio sample a response emits
passes through a per-response causal stream first. Text, tools, reference
audio, the acoustic-history cache and input audio are untouched. The stream
must conserve samples on normal completion and discard its tail on
cancellation or failure; a stream error fails the response closed, so plain
audio never leaks out around the watermark.

``audioseal_factory`` builds the stream the demo runs: AudioSeal's 16 kHz
streaming generator applied as a residual on 24 kHz speech
(``audioseal_stream``), carrying a fixed 16-bit experiment tag.
"""
from collections import deque
from functools import wraps
import time

import numpy as np

# Public experiment version tag, never an account, user or voice identifier.
PAYLOAD = 0xF601


def install(engine_class, factory, report=None):
    original = engine_class.respond

    @wraps(original)
    def respond(self, items, settings, emit, abort, *, session_id):
        if "audio" not in settings["output_modalities"]:
            return original(self, items, settings, emit, abort, session_id=session_id)
        stream = factory()
        pending_marks = deque()
        received = emitted = 0
        timings = []
        first_output = None
        flush_ms = None
        started = time.monotonic()
        status = "failed"

        def publish(pcm):
            nonlocal emitted, first_output
            pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
            if not np.isfinite(pcm).all():
                raise ValueError("Nonfinite watermarked PCM")
            if abort.is_set():
                raise InterruptedError("Response cancelled during watermarking")
            if len(pcm):
                if emitted + len(pcm) > received:
                    raise ValueError("Watermark produced extra samples")
                emit("audio", pcm)
                emitted += len(pcm)
                if first_output is None:
                    first_output = time.monotonic() - started
            while pending_marks and emitted >= pending_marks[0][0]:
                _, mark = pending_marks.popleft()
                emit("chunk", mark)

        def filtered(kind, value):
            nonlocal received
            if abort.is_set():
                raise InterruptedError("Response cancelled before watermarking")
            if kind == "audio":
                # Own the input: the producer may reuse its buffers after emit.
                pcm = np.array(value, dtype=np.float32, copy=True).reshape(-1)
                received += len(pcm)
                begin = time.perf_counter_ns()
                out = stream.push(pcm)
                timings.append((time.perf_counter_ns() - begin) / 1e6)
                publish(out)
            elif kind == "chunk":
                # A phrase mark follows the phrase's last original sample.
                # Keep that order while the causal stream holds a tail.
                pending_marks.append((received, value))
                if emitted >= received:
                    publish(np.empty(0, dtype=np.float32))
            else:
                emit(kind, value)

        try:
            result = original(self, items, settings, filtered, abort, session_id=session_id)
            if abort.is_set():
                raise InterruptedError("Response cancelled before watermark flush")
            begin = time.perf_counter_ns()
            tail = stream.finish()
            flush_ms = (time.perf_counter_ns() - begin) / 1e6
            publish(tail)
            if emitted != received or pending_marks:
                raise ValueError("Watermark did not conserve response samples")
            status = "completed"
            return result
        finally:
            if abort.is_set():
                status = "cancelled"
            stream.cancel()
            if report is not None:
                report({"status": status, "input_samples": received,
                        "output_samples": emitted, "first_output_s": first_output,
                        "chunk_ms": timings, "flush_ms": flush_ms,
                        "pending_marks": len(pending_marks)})

    engine_class.respond = respond


class PassThrough:
    def push(self, pcm):
        return pcm

    def finish(self):
        return np.empty(0, dtype=np.float32)

    def cancel(self):
        pass


def audioseal_factory(checkpoint, payload=PAYLOAD):
    """Load AudioSeal's streaming generator and return a per-response factory.

    Needs ``torch``, ``scipy`` and ``audioseal`` installed; the checkpoint is
    AudioSeal's ``generator_streaming.pth`` (16 bits, 320-sample frames).
    The launcher pins torch to one CPU thread before this runs.
    """
    from audioseal import AudioSeal

    from .audioseal_stream import ResidualResponse, WatermarkPool

    model = AudioSeal.load_generator(str(checkpoint), nbits=16).eval().cpu()
    if model.frame_size != 320:
        raise RuntimeError(f"Unexpected AudioSeal frame size {model.frame_size}")
    pool = WatermarkPool(model, compact=True)
    bits = [int(bit) for bit in f"{payload:016b}"]

    class Stream:
        def __init__(self):
            self.response = ResidualResponse(pool, bits)

        def push(self, pcm):
            return self.response.push(pcm).numpy().reshape(-1)

        def finish(self):
            return self.response.finish().numpy().reshape(-1)

        def cancel(self):
            self.response.cancel()

    # Warm once at startup so no user's first reply pays for it.
    warm = Stream()
    warm.push(np.zeros(1920, dtype=np.float32))
    warm.finish()
    warm.cancel()
    return Stream
