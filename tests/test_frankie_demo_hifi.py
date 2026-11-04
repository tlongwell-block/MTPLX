import os
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
from scipy.signal import resample_poly

from mtplx.frankie.demo.hifi import Stream, install

WEIGHTS = Path(os.environ.get("MTPLX_FRANKIE_HIFI_TEST_WEIGHTS",
                              "~/experiments/frankie-demo-assets/hifi")).expanduser()


class Repeat:
    """Stand-in extender: each input sample twice, so positions are checkable."""

    def __call__(self, pcm):
        return np.repeat(np.asarray(pcm, np.float32), 2)


def respond(body, rate):
    class Engine:
        def respond(self, items, settings, emit, abort, *, session_id):
            return body(emit, abort)

    install(Engine, Repeat())
    events = []
    result = Engine().respond([], {"output_modalities": ["audio"], "output_rate": rate},
                              lambda kind, value: events.append((kind, value)),
                              threading.Event(), session_id="t")
    return result, events


@pytest.mark.parametrize("sizes", [[1920] * 7, [1, 2, 3, 5000, 7], [500]])
def test_stream_conserves_and_orders_samples(sizes):
    x = np.arange(sum(sizes), dtype=np.float32)
    s, out, i = Stream(Repeat()), [], 0
    for n in sizes:
        out.append(s.push(x[i:i + n]))
        i += n
    out.append(s.finish())
    np.testing.assert_array_equal(np.concatenate(out), np.repeat(x, 2))


def test_24k_responses_pass_through():
    _, events = respond(lambda emit, abort: emit("audio", np.arange(5, dtype=np.float32)), 24000)
    assert len(events) == 1 and len(events[0][1]) == 5


def test_48k_marks_follow_their_phrase_without_waiting_for_the_next():
    def body(emit, abort):
        emit("audio", np.ones(3000, np.float32))
        emit("chunk", {"name": "a"})
        emit("audio", np.ones(1000, np.float32))
        emit("chunk", {"name": "b"})
        return "done"

    result, events = respond(body, 48000)
    kinds = [k for k, _ in events]
    assert result == "done" and kinds[-1] == "chunk"
    first_mark = kinds.index("chunk")
    assert sum(len(v) for k, v in events[:first_mark] if k == "audio") == 6000
    assert sum(len(v) for k, v in events if k == "audio") == 8000


def test_cancel_discards_the_held_tail():
    def body(emit, abort):
        emit("audio", np.ones(3000, np.float32))
        abort.set()
        emit("audio", np.ones(10, np.float32))

    with pytest.raises(InterruptedError):
        respond(body, 48000)


@pytest.mark.skipif(not (WEIGHTS / "g_24kto48k.safetensors").is_file(), reason="AP-BWE weights")
def test_real_extender_streams_like_a_whole_render_and_keeps_the_band_below():
    from scipy.signal import firwin, filtfilt

    from mtplx.frankie.demo.hifi import Extender

    rng = np.random.default_rng(0)
    t = np.arange(24000 * 2) / 24000
    # Voiced-ish: harmonics of a gliding pitch plus a little noise, band-limited like Breeze.
    f0 = 180 + 40 * np.sin(2 * np.pi * 0.7 * t)
    x = sum(np.sin(2 * np.pi * np.cumsum(f0 * k) / 24000) / k for k in range(1, 40))
    x = (0.1 * x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)
    x = filtfilt(firwin(255, 11000, fs=24000), [1.0], x).astype(np.float32)
    extend = Extender(WEIGHTS)
    whole = extend(x)
    s = Stream(extend)
    streamed = np.concatenate([s.push(x[i:i + 1920]) for i in range(0, len(x), 1920)] + [s.finish()])
    assert len(streamed) == len(whole) == 2 * len(x)
    err = np.sum((streamed - whole) ** 2) / np.sum(whole ** 2)
    assert 10 * np.log10(err) < -60
    # Below the crossover the output is the input, upsampled.
    low = firwin(511, 9000, fs=48000)
    up = resample_poly(x, 2, 1)
    d = filtfilt(low, [1.0], whole - up)[2000:-2000]
    assert 10 * np.log10(np.sum(d ** 2) / np.sum(filtfilt(low, [1.0], up)[2000:-2000] ** 2)) < -60
