"""Native interleaving filters private tokens and hands public speech off early."""

from threading import Event
from types import SimpleNamespace as NS

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def cpu_and_eager(monkeypatch):
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    try:
        yield
    finally:
        mx.set_default_device(device)


def run_voice(monkeypatch, pieces, *, mtp, thinking="off", termination="stop", inspect=None):
    from mtplx.frankie import engine as module

    spoken, events, resets = [], [], []
    abort = Event()
    values = dict(enumerate(pieces, start=2))

    class Detokenizer:
        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = values[token]

        def finalize(self):
            self.last_segment = ""

    def speak(text, states):
        spoken.append((text, states.tolist()))
        yield mx.zeros((960,))

    def generate(runtime, ids, *, token_callback, **kwargs):
        assert kwargs["max_tokens"] == 32
        assert kwargs["thinking_guard"] is guard
        stream = runtime.model._mtplx_feature_stream
        for index, token in enumerate(values):
            # Each actual committed token has a unique feature. Private tokens
            # must consume their own rows without shifting later public speech.
            position = len(ids) + index
            stream.record(mx.array([[token]]), mx.full((1, 1, 4), float(token)),
                          [NS(offset=position + 1)], position)
            token_callback([token])
            if inspect:
                inspect(index, spoken)
        if termination == "cancel":
            abort.set()
            token_callback([])
        if termination == "error":
            raise RuntimeError("Synthetic inference failure")
        return NS(stats=NS(to_dict=dict), finish_reason=termination)

    guard = object()
    monkeypatch.setattr(module, "thinking_guard", lambda *a: guard)
    monkeypatch.setattr(module, "generate_mtpk", generate)
    monkeypatch.setattr(module, "generate_ar", generate)
    engine = module.Frankie.__new__(module.Frankie)
    engine.mtp = mtp
    engine.runtime = NS(model=NS(model=NS(layers=[None] * 20)))
    engine.bank = None
    engine.audio = NS(speak=speak, reset_speech_context=lambda: resets.append(True))
    engine.tokenizer = NS(detokenizer=Detokenizer(), eos_token_ids=[0],
                          decode=lambda tokens: "".join(values[token] for token in tokens))
    engine.prompt = lambda *a, **kw: ([1], None)
    settings = {"output_modalities": ["audio"], "max_output_tokens": 32, "thinking": thinking}
    result = None
    if termination == "error":
        with pytest.raises(RuntimeError, match="Synthetic inference failure"):
            engine.respond([], settings, lambda *event: events.append(event), abort,
                           session_id="native-thinking")
    else:
        result = engine.respond([], settings, lambda *event: events.append(event), abort,
                                session_id="native-thinking")
    assert engine.runtime.model._mtplx_feature_stream is None
    assert len(resets) == 2
    return result, spoken, events


@pytest.mark.parametrize("mtp", [0, 2, 3])
def test_public_ack_and_updates_are_spoken_before_each_native_private_block(monkeypatch, mtp):
    pieces = ["</think>", "Let me check.", "<think>", "PRIVATE ONE", "</think>",
              "I found the details.", "<think>", "PRIVATE TWO", "</think>", "Here is the answer."]

    def inspect(index, spoken):
        if index == 2:
            assert [text for text, _ in spoken] == ["Let me check."]
        elif index == 6:
            assert [text for text, _ in spoken] == ["Let me check.", "I found the details."]

    result, spoken, events = run_voice(monkeypatch, pieces, mtp=mtp, thinking="high", inspect=inspect)
    assert spoken == [
        ("Let me check.", [[3.0] * 4]),
        ("I found the details.", [[7.0] * 4]),
        ("Here is the answer.", [[11.0] * 4]),
    ]
    assert result["raw_text"] == "".join(pieces)
    assert result["text"] == "Let me check.I found the details.Here is the answer."
    assert "PRIVATE" not in "".join(value for kind, value in events if kind == "text")
    assert [chunk["text"] for chunk in result["chunks"]] == [text for text, _ in spoken]


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("termination", ["stop", "length", "cancel", "error"])
@pytest.mark.parametrize("acknowledge", [False, True])
def test_private_tail_never_becomes_speech_on_any_termination(monkeypatch, mtp, termination, acknowledge):
    prefix = ["</think>", "Let me check.", "<think>"] if acknowledge else []
    result, spoken, events = run_voice(
        monkeypatch, prefix + ["PRIVATE UNFINISHED"], mtp=mtp,
        thinking="high", termination=termination)
    assert [text for text, _ in spoken] == (["Let me check."] if acknowledge else [])
    public = "".join(value for kind, value in events if kind == "text")
    assert public == ("Let me check." if acknowledge else "")
    if result is not None:
        assert result["text"] == public and result["finish_reason"] == termination


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("thinking,prefix", [("off", []), ("high", ["</think>"])])
def test_ordinary_direct_answers_keep_their_public_feature_rows(monkeypatch, mtp, thinking, prefix):
    result, spoken, _events = run_voice(
        monkeypatch, prefix + ["Hello.", " Welcome."], mtp=mtp, thinking=thinking)
    assert result["text"] == "Hello. Welcome."
    assert [text for text, _ in spoken] == ["Hello.", "Welcome."]
    assert [rows for _, rows in spoken] == [
        [[float(2 + len(prefix))] * 4], [[float(3 + len(prefix))] * 4]]
