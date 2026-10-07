import threading
from contextlib import nullcontext

import numpy as np
import pytest

from mtplx.frankie.demo.watermark import PassThrough, install


class Delayed:
    """A causal stream that always holds back its last three samples."""

    def __init__(self):
        self.pending = np.empty(0, dtype=np.float32)
        self.closed = False

    def push(self, x):
        self.pending = np.concatenate((self.pending, x))
        n = max(0, len(self.pending) - 3)
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def finish(self):
        out, self.pending = self.pending, self.pending[:0]
        return out

    def cancel(self):
        self.closed = True
        self.pending = self.pending[:0]


def respond(factory, body, *, audio=True):
    class Engine:
        def respond(self, items, settings, emit, abort, *, session_id):
            return body(emit, abort)

    rows, events = [], []
    install(Engine, factory, rows.append)
    result = Engine().respond(
        [], {"output_modalities": ["audio"] if audio else ["text"]},
        lambda kind, value: events.append((kind, value)), threading.Event(),
        session_id="test")
    return result, rows, events


def test_phrase_mark_follows_its_audio_and_tools_are_not_held():
    def body(emit, abort):
        emit("audio", np.arange(5, dtype=np.float32))
        emit("chunk", {"name": "a"})
        emit("tool_call", {"name": "tool"})
        emit("audio", np.arange(5, 9, dtype=np.float32))
        emit("chunk", {"name": "b"})
        return {"text": "complete"}

    result, rows, events = respond(Delayed, body)
    assert result == {"text": "complete"}
    assert [k for k, _ in events] == ["audio", "tool_call", "audio", "chunk", "audio", "chunk"]
    np.testing.assert_array_equal(
        np.concatenate([v for k, v in events if k == "audio"]), np.arange(9))
    assert rows[0]["status"] == "completed"
    assert rows[0]["input_samples"] == rows[0]["output_samples"] == 9


def test_cancellation_during_transform_discards_tail():
    streams, abort = [], []

    class Stop(Delayed):
        def push(self, x):
            out = super().push(x)
            abort[0].set()
            return out

    def factory():
        streams.append(Stop())
        return streams[-1]

    def body(emit, event):
        abort.append(event)
        emit("audio", np.ones(10))

    with pytest.raises(InterruptedError):
        respond(factory, body)
    assert streams[0].closed and len(streams[0].pending) == 0


def test_stream_error_emits_no_plain_audio():
    class Bad(Delayed):
        def push(self, x):
            raise ValueError("model failed")

    events = []

    class Engine:
        def respond(self, items, settings, emit, abort, *, session_id):
            emit("audio", np.ones(10))

    install(Engine, Bad)
    with pytest.raises(ValueError, match="model failed"):
        Engine().respond([], {"output_modalities": ["audio"]},
                         lambda k, v: events.append(k), threading.Event(), session_id="t")
    assert events == []


@pytest.mark.parametrize("n", [1, 2, 17])
def test_responses_do_not_share_a_tail(n):
    _, _, events = respond(Delayed, lambda emit, abort: emit("audio", np.ones(n)))
    assert sum(len(v) for k, v in events if k == "audio") == n


@pytest.mark.parametrize("output", [np.array([np.nan]), np.ones(50)])
def test_nonfinite_or_extra_output_is_rejected(output):
    class Bad(Delayed):
        def push(self, x):
            return output

    with pytest.raises(ValueError):
        respond(Bad, lambda emit, abort: emit("audio", np.ones(4)))


def test_text_only_never_builds_a_stream():
    def factory():
        raise AssertionError("constructed")

    _, rows, events = respond(factory, lambda emit, abort: emit("text", "hello"), audio=False)
    assert rows == [] and events == [("text", "hello")]


def test_pass_through_preserves_sample_values():
    _, _, events = respond(
        PassThrough, lambda emit, abort: emit("audio", np.arange(17, dtype=np.float32)))
    np.testing.assert_array_equal(events[0][1], np.arange(17))


class Unmarked:
    """AudioSeal's streaming interface with a watermark of exactly zero."""

    frame_size = 320

    def streaming(self, batch_size):
        return nullcontext()

    def set_streaming_state(self, state):
        pass

    def get_streaming_state(self):
        return {}

    def __call__(self, x, message):
        assert x.shape[-1] % self.frame_size == 0
        return x.clone()


@pytest.mark.parametrize("sizes", [[1920], [7, 500, 1, 2400], [5000, 3]])
def test_residual_stream_conserves_speech_bit_exact(sizes):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from mtplx.frankie.demo.audioseal_stream import ResidualResponse, WatermarkPool

    pcm = np.random.default_rng(0).uniform(-0.5, 0.5, sum(sizes)).astype(np.float32)
    stream = ResidualResponse(WatermarkPool(Unmarked(), compact=True), [1, 0] * 8)
    out, start = [], 0
    for n in sizes:
        out.append(stream.push(pcm[start:start + n]).numpy().reshape(-1))
        start += n
    out.append(stream.finish().numpy().reshape(-1))
    np.testing.assert_array_equal(np.concatenate(out), pcm)


def test_residual_stream_rejects_a_wrong_payload():
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from mtplx.frankie.demo.audioseal_stream import ResidualResponse, WatermarkPool

    with pytest.raises(ValueError):
        ResidualResponse(WatermarkPool(Unmarked()), [2] * 16)
