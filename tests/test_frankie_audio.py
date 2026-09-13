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


@pytest.mark.parametrize("temperature", [0, 0.9])
@pytest.mark.parametrize("top_p", [0.8, 1.0])
@pytest.mark.parametrize("top_k", [0, 20, 50])
@pytest.mark.parametrize("seed", [3, 19])
def test_breeze_depth_matches_upstream_decoder(temperature, top_p, top_k, seed):
    from mlx import nn
    from mlx_audio.tts.models.breeze_tts.breeze_tts import _DepthModel

    from mtplx.frankie.breeze import BreezeModel, Model, ModelConfig

    # The independent upstream path recomputes the full prefix and samples
    # scalar tokens; exercise the real attention and sampling math in both paths.
    mx.random.seed(0)
    config = ModelConfig(
        num_codebooks=8,
        vocab_size=35,
        depth_decoder_config={
            "num_codebooks": 8,
            "vocab_size": 35,
            "audio_embed_size": 32,
            "backbone_hidden_size": 32,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
        },
    )
    model = BreezeModel.__new__(BreezeModel)
    model.config, model.vocab_size, model.num_codebooks = config, 32, 8
    model.depth_heads = [nn.Linear(32, 35, bias=False) for _ in range(7)]
    depth = _DepthModel(config)
    model.depth_decoder = NS(
        model=depth,
        next_logits=lambda ids, h: model.depth_heads[ids.shape[1] - 2](
            depth(ids, h)[:, -1]
        ),
    )
    hidden = mx.random.normal((1, 32))
    mx.eval(hidden, model.depth_heads, depth.parameters())
    settings = {
        "unconditional_hidden": None,
        "cfg_scale": 1,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    mx.random.seed(seed)
    expected = Model._depth_tokens(model, 1, hidden, **settings)
    mx.random.seed(seed)
    actual = model._depth_tokens(1, hidden, **settings)
    assert actual == expected
    assert len(actual) == 8 and all(0 <= token < 32 for token in actual)


@pytest.fixture
def breeze_context(monkeypatch):
    from mtplx.frankie.breeze import BreezeModel, Model

    model = BreezeModel.__new__(BreezeModel)
    model.context_rows = 2048
    model.context_words = 100
    model._voice_prefix = mx.zeros((200, 4))
    model.backbone_model = NS(embed_tokens=lambda ids: mx.zeros((1, 1, 4)))
    model._new_cache = lambda: [NS(offset=0, keys=NS(nbytes=1), values=NS(nbytes=1))]
    model.config = NS(codebook_eos_token_id=0)
    model.num_codebooks = 1
    model.reset_speech_context()
    monkeypatch.setattr(
        Model, "_prompt_embeddings", lambda *a, **kw: mx.zeros((1, 3, 4))
    )

    def generate(self, *args, **kwargs):
        prompt = self._prompt_embeddings(*args, **kwargs)
        cache = self._generation_cache()
        cache[0].offset += prompt.shape[1] + 40
        yield NS(token_count=40)

    monkeypatch.setattr(Model, "generate", generate)
    return model


def test_breeze_retained_context_has_word_and_row_limits(breeze_context):
    model = breeze_context
    list(model.generate("word " * 60, max_tokens=75))
    first = model._speech_cache
    list(model.generate("word " * 40, max_tokens=75))
    assert model._speech_cache is first
    assert model._context_words == 100
    list(model.generate("Next.", max_tokens=75))
    assert model._speech_cache is not first
    assert model._context_words == 1

    first = model._speech_cache
    first[0].offset = model.context_rows - 50
    list(model.generate("New sentence.", max_tokens=75))
    assert model._speech_cache is not first
    assert model._context_words == 2


@pytest.mark.parametrize(
    "case", ["long_text", "long_reference", "large_cache", "disabled", "cancel"]
)
def test_breeze_discards_unusable_context(breeze_context, case):
    model = breeze_context
    if case == "long_reference":
        model._voice_prefix = mx.zeros((2100, 4))
    if case == "large_cache":
        model._new_cache = lambda: [
            NS(offset=0, keys=NS(nbytes=50_000_001), values=NS(nbytes=50_000_000))
        ]
    if case == "disabled":
        model.context_words = 0
    generator = model.generate("word " * (101 if case == "long_text" else 1))
    if case == "cancel":
        next(generator)
        generator.close()
    else:
        assert list(generator)
    assert model._speech_cache is None
    assert model._context_words == 0


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
    context_resets = []
    engine.audio = NS(
        speak=speak, reset_speech_context=lambda: context_resets.append(True)
    )
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
    assert len(context_resets) == 2
    assert engine.runtime.model._mtplx_feature_stream is None
