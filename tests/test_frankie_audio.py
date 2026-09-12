from types import SimpleNamespace as NS

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("soundfile")
pytest.importorskip("scipy")
pytest.importorskip("parakeet_mlx")

from mtplx.frankie.audio import AudioModels
from mtplx.frankie.bridges import EarBridge


def test_shared_ctc_logits_preserve_ear_bridge_output():
    bridge = EarBridge(4, 3, hidden=4)
    frames = mx.random.normal((8, 512))
    direct = bridge(frames)
    shared = bridge(frames, ctc=bridge.ctc(frames))
    assert mx.array_equal(direct, shared).item()


def test_hearing_and_transcription_share_encoder_and_ctc(monkeypatch):
    import parakeet_mlx.audio

    monkeypatch.setattr(parakeet_mlx.audio, "get_logmel", lambda *a: mx.zeros((32, 4)))
    calls = []
    frames = mx.zeros((1, 8, 512))

    def encoder(mel):
        calls.append("encoder")
        return frames, mx.array([8])

    def ctc(x):
        calls.append("ctc")
        return 10 * mx.eye(4)[mx.array([0, 0, 3, 0, 1, 1, 3, 2])]

    audio = AudioModels.__new__(AudioModels)
    audio.ear = NS(
        encoder=encoder, preprocessor_config=None, vocabulary=["A", "B", "!"]
    )
    audio.bridge = EarBridge(4, 3, hidden=4)
    audio.bridge.ctc = ctc
    audio.tone = lambda x: mx.zeros((1, 3))
    rows, transcript = audio.hear(mx.zeros((2400,)))
    assert rows.shape == (9, 3)
    assert rows.dtype == mx.bfloat16
    assert transcript == "AAB!"
    assert calls == ["encoder", "ctc"]


def test_prepared_audio_is_reused_by_conversation_history():
    from mtplx.frankie.engine import Frankie

    calls = []
    rows = object()

    def hear(pcm, rate):
        calls.append(rate)
        return rows, "Hello."

    engine = Frankie.__new__(Frankie)
    engine.audio = NS(hear=hear)
    item = {"id": "input", "content": [{"type": "input_audio", "_pcm": object()}]}
    assert engine.prepare_audio(item) == [(item, 0, "Hello.")]
    assert engine.prepare_audio(item) == [(item, 0, "Hello.")]
    assert item["content"][0]["_rows"] is rows
    assert calls == [24000]


def test_cancelled_queued_response_does_not_prepare_audio():
    from threading import Event

    from mtplx.frankie.engine import Frankie

    engine = Frankie.__new__(Frankie)
    abort = Event()
    abort.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        engine.respond([], {}, lambda *a: None, abort, session_id="cancelled")


@pytest.mark.parametrize("mtp", [0, 3])
def test_cancel_during_playback_unwinds_decode_and_features(monkeypatch, mtp):
    from threading import Event

    import mtplx.thinking_guard
    from mtplx.frankie import engine as module

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setattr(mtplx.thinking_guard, "think_marker_ids", lambda _: None)
    abort = Event()
    closed = []

    class Detokenizer:
        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = {2: "Hello.", 3: " Next"}[token]

    def speak(*args):
        try:
            yield mx.zeros((2400,))
            abort.set()
            yield mx.zeros((2400,))
        finally:
            closed.append(True)

    def generate(runtime, ids, *, token_callback, **kwargs):
        stream = runtime.model._mtplx_feature_stream
        stream.record(mx.array([[2, 3]]), mx.ones((1, 2, 4)), [NS(offset=3)], 1)
        token_callback([2, 3])
        pytest.fail("Cancellation must stop decoding at the feature callback.")

    monkeypatch.setattr(module, "generate_ar", generate)
    monkeypatch.setattr(module, "generate_mtpk", generate)
    engine = module.Frankie.__new__(module.Frankie)
    engine.mtp = mtp
    engine.runtime = NS(model=NS(model=NS(layers=[None] * 20)))
    engine.bank = None
    engine.tokenizer = NS(
        detokenizer=Detokenizer(),
        eos_token_ids=[0],
        decode=lambda ids: "".join({2: "Hello.", 3: " Next"}[i] for i in ids),
    )
    engine.audio = NS(speak=speak)
    engine.prompt = lambda *a, **kw: ([1], None)
    emitted = []
    with pytest.raises(InterruptedError, match="cancelled"):
        engine.respond(
            [],
            {"max_output_tokens": 512, "output_modalities": ["audio"]},
            lambda kind, value: emitted.append(kind),
            abort,
            session_id="cancelled",
        )
    assert emitted.count("audio") == 1
    assert closed == [True]
    assert engine.runtime.model._mtplx_feature_stream is None
