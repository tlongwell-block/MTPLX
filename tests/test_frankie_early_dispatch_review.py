"""Independent native-token admission checks with real feature commitment on CPU."""

import json
from threading import Event
from types import SimpleNamespace as NS

import pytest

mx = pytest.importorskip("mlx.core")

OPEN, CLOSE = "<tool_call>", "</tool_call>"
BODY = '{"name":"lookup","arguments":{"topic":"weather"}}'
TOOLS = [{"type": "function", "name": "lookup", "description": "Fixture lookup",
          "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}}}]


@pytest.fixture(autouse=True)
def cpu_eager(monkeypatch):
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    try:
        yield
    finally:
        mx.set_default_device(device)


def engine_run(monkeypatch, pieces, *, mtp=2, audio=True, background=True,
               rejected=False, inspect=None, thinking="off", batch=False):
    from mtplx.frankie import engine as module

    values = dict(enumerate(pieces, 2))
    values.update({900: OPEN, 901: BODY, 902: CLOSE})
    committed = list(range(2, len(pieces)+2))
    emitted, spoken = [], []

    class Detokenizer:
        def reset(self):
            self.last_segment = ""
        def add_token(self, token):
            self.last_segment = values[token]
        def finalize(self):
            self.last_segment = ""

    def speak(text, states):
        spoken.append(text)
        yield mx.zeros((960,))

    def generate(runtime, ids, *, token_callback, **kwargs):
        stream = getattr(runtime.model, "_mtplx_feature_stream", None)
        if rejected:
            assert stream is not None
            # A full tempting tool envelope was evaluated speculatively, but
            # never committed. It must not become tool work or public text.
            stream.record(mx.array([[900, 901, 902]]), mx.ones((1, 3, 4)),
                          [NS(offset=len(ids)+3)], len(ids))
        if batch:
            if stream:
                stream.record(mx.array([committed]), mx.ones((1, len(committed), 4)),
                              [NS(offset=len(ids)+len(committed))], len(ids))
            token_callback(committed)
            if inspect:
                inspect(len(committed)-1, emitted)
        else:
            for index, token in enumerate(committed):
                if stream:
                    stream.record(mx.array([[token]]), mx.full((1, 1, 4), float(token)),
                                  [NS(offset=len(ids)+index+1)], len(ids)+index)
                token_callback([token])
                if inspect:
                    inspect(index, emitted)
        return NS(stats=NS(to_dict=dict), finish_reason="stop")

    monkeypatch.setattr(module, "thinking_guard", lambda *args: None)
    monkeypatch.setattr(module, "generate_mtpk", generate)
    monkeypatch.setattr(module, "generate_ar", generate)
    engine = module.Frankie.__new__(module.Frankie)
    engine.mtp, engine.bank = mtp, None
    engine.runtime = NS(model=NS(model=NS(layers=[None]*20)))
    engine.audio = NS(reset_speech_context=lambda: None, speak=speak)
    engine.tokenizer = NS(detokenizer=Detokenizer(), eos_token_ids=[0],
                          decode=lambda tokens: "".join(values[token] for token in tokens))
    engine.prompt = lambda *args, **kwargs: ([1], None)
    result = engine.respond([], {"output_modalities": ["audio"] if audio else ["text"],
                                "max_output_tokens": 64, "thinking": thinking,
                                "background_tasks": background, "tools": TOOLS},
                            lambda *event: emitted.append(event), Event(), session_id="review")
    assert getattr(engine.runtime.model, "_mtplx_feature_stream", None) is None
    return result, emitted, spoken


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("audio", [False, True])
def test_public_committed_close_emits_once_before_generator_finishes(monkeypatch, mtp, batch, audio):
    def inspect(index, events):
        calls = [value for kind, value in events if kind == "tool_call"]
        if index < 3:
            assert not calls
        else:
            assert len(calls) == 1 and calls[0]["key"] == (4, 0)
    result, events, spoken = engine_run(monkeypatch, ["I can check.", OPEN, BODY, CLOSE, " One moment."],
                                        mtp=mtp, audio=audio, batch=batch, inspect=inspect)
    calls = [value for kind, value in events if kind == "tool_call"]
    assert len(calls) == 1 and result["tool_events"] == calls
    assert result["tool_events"][0]["call"]["id"] == calls[0]["call"]["id"]
    assert json.loads(calls[0]["call"]["function"]["arguments"]) == {"topic": "weather"}
    assert all("weather" not in text and OPEN not in text for text in spoken)
    assert result["text"] == "I can check. One moment."


@pytest.mark.parametrize("pieces", [
    ["<think>", OPEN, BODY, CLOSE, "</think>"],
    [OPEN, BODY],
    [OPEN, OPEN, BODY, CLOSE, CLOSE],
    [OPEN, '{"name":"lookup","arguments":{"topic":"', "<think>", "PRIVATE", "</think>", 'weather"}}', CLOSE],
    [OPEN, BODY, "<think>", "PRIVATE UNFINISHED", CLOSE],
])
def test_private_incomplete_nested_and_private_split_envelopes_never_dispatch(monkeypatch, pieces):
    result, events, _ = engine_run(monkeypatch, pieces)
    assert not [value for kind, value in events if kind == "tool_call"]
    assert not result.get("tool_events")


def test_public_call_after_private_reasoning_is_eligible_but_private_call_is_not(monkeypatch):
    result, events, _ = engine_run(monkeypatch, ["<think>", OPEN, BODY, CLOSE, "</think>", OPEN, BODY, CLOSE])
    calls = [value for kind, value in events if kind == "tool_call"]
    assert len(calls) == 1 and calls == result["tool_events"]
    assert calls[0]["key"] == (8, 0)


def test_evaluated_rejected_tool_rows_never_dispatch_without_commit(monkeypatch):
    result, events, spoken = engine_run(monkeypatch, ["Hello."], rejected=True)
    assert not result.get("tool_events")
    assert not [value for kind, value in events if kind == "tool_call"]
    assert spoken == ["Hello."] and result["raw_text"] == "Hello."


def test_no_background_mode_keeps_raw_terminal_fallback_without_early_calls(monkeypatch):
    result, events, spoken = engine_run(monkeypatch, [OPEN, BODY, CLOSE], background=False)
    assert not [value for kind, value in events if kind == "tool_call"]
    assert "tool_events" not in result
    assert result["raw_text"] == OPEN + BODY + CLOSE and not spoken


def test_distinct_identical_native_envelopes_have_distinct_positions_and_ids(monkeypatch):
    result, events, _ = engine_run(monkeypatch, [OPEN, BODY, CLOSE, OPEN, BODY, CLOSE])
    calls = [value for kind, value in events if kind == "tool_call"]
    assert calls == result["tool_events"] and [call["key"] for call in calls] == [(3, 0), (6, 0)]
    assert len({call["call"]["id"] for call in calls}) == 2
