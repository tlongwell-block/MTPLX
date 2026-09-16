"""Text-only realtime output does not need hidden speech features."""

from threading import Event
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("mtp", [0, 2])
def test_text_stream_uses_committed_tokens_without_speech_capture(monkeypatch, mtp):
    from mtplx.frankie import engine as module

    pieces = {2: "<think>", 3: "private", 4: "</think>", 5: "Hello.", 6: " Welcome."}

    class Detokenizer:
        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = pieces[token]

        def finalize(self):
            self.last_segment = ""

    def no_features(*args, **kwargs):
        pytest.fail("Text-only output must not capture speech states")

    monkeypatch.setattr(module, "CommittedFeatures", no_features)
    monkeypatch.setattr(module, "thinking_guard", lambda *a: None)

    def generate(runtime, ids, *, token_callback, **kwargs):
        assert ids == [1]
        token_callback([2, 3, 4, 5])
        token_callback([6])
        return NS(stats=NS(to_dict=dict), finish_reason="stop")

    monkeypatch.setattr(module, "generate_mtpk", generate)
    monkeypatch.setattr(module, "generate_ar", generate)
    engine = module.Frankie.__new__(module.Frankie)
    engine.mtp = mtp
    engine.runtime = NS()
    engine.bank = None
    engine.audio = NS(reset_speech_context=lambda: None, speak=no_features)
    engine.tokenizer = NS(
        detokenizer=Detokenizer(),
        eos_token_ids=[0],
        decode=lambda ids: "".join(pieces[i] for i in ids),
    )
    engine.prompt = lambda *a, **kw: ([1], None)
    progress = []
    engine.background_step = progress.append
    emitted = []
    result = engine.respond(
        [],
        {
            "output_modalities": ["text"],
            "max_output_tokens": 32,
            "thinking": "off",
            "temperature": 0,
        },
        lambda *event: emitted.append(event),
        Event(),
        session_id="text",
    )
    assert emitted == [("text", "Hello."), ("text", " Welcome.")]
    assert result["text"] == "Hello. Welcome."
    assert result["raw_text"] == "<think>private</think>Hello. Welcome."
    assert result["audio_seconds"] == 0
    assert result["chunks"] == []
    assert progress == [float("inf"), float("inf")]
