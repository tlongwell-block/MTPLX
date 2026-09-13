"""HTTP and realtime share brain policy without overriding explicit sampling."""

from types import SimpleNamespace as NS

import pytest

from mtplx.frankie.sampling import brain_sampler, thinking_mode


@pytest.mark.parametrize("mode", ["minimal", "low", "medium", "high", "xhigh", "max"])
def test_thinking_defaults_follow_mode_and_preserve_overrides(mode):
    http = brain_sampler({"reasoning_effort": mode})
    voice = brain_sampler({"thinking": mode}, realtime=True)
    assert http == voice
    assert http.temperature == 1 and http.top_p == .95 and http.top_k == 20
    assert http.presence_penalty == 0
    assert brain_sampler({"thinking": mode, "temperature": 0}).temperature == 0
    assert brain_sampler({"thinking": mode, "temperature": None}) == http


def test_non_thinking_keeps_voice_penalties_independent():
    http = brain_sampler({"reasoning_effort": "none"})
    voice = brain_sampler({}, realtime=True)
    assert http.temperature == voice.temperature == .7
    assert http.top_p == voice.top_p == .8
    assert http.presence_penalty == 1.5 and voice.presence_penalty == 0
    assert brain_sampler({"presence_penalty": 0}).presence_penalty == 0
    assert thinking_mode({"enable_thinking": True}) != "off"
    with pytest.raises(ValueError, match="conflicts"):
        thinking_mode({"reasoning_effort": "none", "enable_thinking": True})


@pytest.mark.parametrize("settings", [{"temperature": -1}, {"temperature": 3},
    {"temperature": float("nan")}, {"top_p": 2}, {"top_k": 1.5},
    {"presence_penalty": 3}, {"frequency_penalty": -3}, {"min_p": .1},
    {"repeat_penalty": 1.1}, {"repetition_penalty": 1.1}])
def test_invalid_or_unsupported_sampling_is_not_silently_ignored(settings):
    with pytest.raises(ValueError):
        brain_sampler(settings)


def test_http_ar_penalties_only_count_generated_tokens():
    import mlx.core as mx
    from mtplx.frankie.completions import Completions

    class Detokenizer:
        def reset(self):
            pass

    service = Completions.__new__(Completions)
    service.context_tokens = 64
    service.engine = NS(mtp=0, tokenizer=NS(detokenizer=Detokenizer()))
    job = NS(chat=False, splice=None, data={"prompt": [0, 1, 2], "max_tokens": 12,
        "thinking": "off", "enable_thinking": False, "presence_penalty": 1.5, "frequency_penalty": .5})
    service.prepare(job)
    logits = mx.zeros((1, 8))
    for processor in job.processors:
        logits = processor(mx.array([0, 1, 2, 4, 4]), logits)
    assert logits.tolist()[0] == [0, 0, 0, 0, -2.5, 0, 0, 0]

    job.data.update(temperature=.5, top_p=.5, top_k=4, seed=42)
    service.prepare(job)
    # Temperature changes the nucleus to a single token. Applying top-p
    # before temperature would leave two candidates and occasionally draw 1.
    logprobs = mx.log(mx.array([[.4, .3, .2, .1]]))
    assert mx.stack([job.sampler(logprobs) for _ in range(64)]).tolist() == [[0]] * 64


def test_tool_parser_supplies_distinct_ids():
    from mtplx.server.omlx_bridge.tool_calling import parse_tool_calls

    raw = '<tool_call>{"name":"lookup","arguments":{"city":"Paris"}}</tool_call>'
    calls = [parse_tool_calls(raw, None).tool_calls[0] for _ in range(2)]
    assert all(call["id"] and call["type"] == "function" for call in calls)
    assert calls[0]["id"] != calls[1]["id"]
