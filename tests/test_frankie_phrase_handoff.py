"""Speech may use a committed boundary before the following feature is ready."""

from threading import Event
from types import SimpleNamespace as NS

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize(
    "phrase,following,early",
    [
        ("Hello.", " Welcome.", True),
        ("Here are four words,", " and more.", True),
        ("Hi,", " there.", False),
        ("The value is 3.", "14", False),
        ("Let me check.", "<tool_call>", True),
        ("word " * 15 + "word", " next.", True),
    ],
)
def test_committed_boundary_keeps_exact_phrase_features(
    monkeypatch, mtp, phrase, following, early
):
    from mtplx.frankie import engine as module

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setattr(module, "thinking_guard", lambda *a: None)
    pieces = {2: phrase, 3: following}
    spoken = []

    class Detokenizer:
        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = pieces[token]

        def finalize(self):
            self.last_segment = ""

    def speak(text, states):
        spoken.append((text, states.tolist()))
        yield mx.zeros((1920,))

    def generate(runtime, ids, *, token_callback, **kwargs):
        stream = runtime.model._mtplx_feature_stream
        # A speculative row is present, but it has not been committed.
        stream.record(
            mx.array([[2, 99]]), mx.full((1, 2, 4), 7.0), [NS(offset=3)], 1
        )
        token_callback([2])
        assert not spoken, "Uncommitted speculation cannot trigger speech"
        token_callback([3])
        assert bool(spoken) is early
        if early:
            assert spoken == [(phrase.strip(), [[7.0] * 4])]
        # The real next row arrives after verification replaces speculation.
        stream.record(mx.array([[3]]), mx.full((1, 1, 4), 9.0), [NS(offset=3)], 2)
        token_callback([])
        return NS(stats=NS(to_dict=dict), finish_reason="stop")

    monkeypatch.setattr(module, "generate_mtpk", generate)
    monkeypatch.setattr(module, "generate_ar", generate)
    engine = module.Frankie.__new__(module.Frankie)
    engine.mtp = mtp
    engine.runtime = NS(model=NS(model=NS(layers=[None] * 20)))
    engine.bank = None
    engine.audio = NS(speak=speak, reset_speech_context=lambda: None)
    engine.tokenizer = NS(
        detokenizer=Detokenizer(),
        eos_token_ids=[0],
        decode=lambda ids: "".join(pieces[i] for i in ids),
    )
    engine.prompt = lambda *a, **kw: ([1], None)
    result = engine.respond(
        [],
        {"output_modalities": ["audio"], "max_output_tokens": 32, "thinking": "off"},
        lambda *a: None,
        Event(),
        session_id="boundary",
    )
    assert result["raw_text"] == phrase + following
    if early:
        assert spoken[0][0] == phrase.strip()
    else:
        assert spoken == [((phrase + following).strip(), [[7.0] * 4, [9.0] * 4])]
