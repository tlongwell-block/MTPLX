"""Cooperative native MTP preserves sampling and bounded request progress."""

import contextvars

import mlx.core as mx
import pytest

from mtplx.generation import (
    PromptState,
    _prefill_committed_mtp_history_streaming,
    generate_mtpk,
)
from mtplx.sampling import SamplerConfig
from test_generation_sustained import AcceptingTinyMTPModel, _runtime


@pytest.fixture(autouse=True)
def cpu_and_eager(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    try:
        yield
    finally:
        mx.set_default_device(previous)


def advance(iterator, context):
    try:
        context.run(next, iterator)
    except StopIteration as done:
        return done.value


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("maximum", [1, 2, 3, 4, 12])
@pytest.mark.parametrize("temperature", [0, 0.7])
def test_interleaved_requests_match_serial_sampling(depth, maximum, temperature):
    runtime = _runtime(AcceptingTinyMTPModel())
    random_before = [value.tolist() for value in mx.random.state]
    options = dict(max_tokens=maximum, speculative_depth=depth,
        sampler=SamplerConfig(temperature=temperature, top_p=1, top_k=0),
        stop_token_ids=set(), mtp_history_policy="committed")
    expected = [generate_mtpk(runtime, [0], seed=seed, **options) for seed in (11, 23)]
    streams = [generate_mtpk.steps(runtime, [0], seed=seed, **options) for seed in (11, 23)]
    contexts = [contextvars.copy_context(), contextvars.copy_context()]
    done = [None, None]
    while any(value is None for value in done):
        for i in range(2):
            if done[i] is None:
                done[i] = advance(streams[i], contexts[i])
    for actual, serial in zip(done, expected):
        assert actual.tokens == serial.tokens
        assert len(actual.tokens) == maximum
        assert actual.stats.drafted_tokens == serial.stats.drafted_tokens
        assert actual.stats.accepted_drafts == serial.stats.accepted_drafts
    assert [value.tolist() for value in mx.random.state] == random_before


def test_prepared_prompt_is_not_evaluated_again():
    model = AcceptingTinyMTPModel()
    runtime = _runtime(model)
    prompt = [0] * 70
    prefill = _prefill_committed_mtp_history_streaming.steps(
        runtime, prompt, prefill_chunk_size=32)
    progress = []
    while True:
        try:
            progress.append(next(prefill)["prefill_tokens"])
        except StopIteration as done:
            cache, logits, hidden, draft, target_s, history_s, position_base = done.value
            break
    assert progress == [32, 64, 69]
    state = PromptState(trunk_cache=cache, logits=logits, hidden=hidden,
        committed_mtp_cache=draft, token_prefix=tuple(prompt),
        prompt_eval_time_s=target_s + history_s, mtp_history_policy="committed",
        mtp_history_position_base=position_base)
    model.calls.clear()
    steps = generate_mtpk.steps(runtime, prompt, _prompt_state=state,
        speculative_depth=3, max_tokens=8, sampler=SamplerConfig(temperature=0),
        mtp_history_policy="committed", stop_token_ids=set())
    next(steps)
    assert model.calls == []
    while advance(steps, contextvars.copy_context()) is None:
        pass
    assert all(call["tokens"] <= 4 for call in model.calls)


def test_admission_does_not_inherit_voice_image_context():
    from types import SimpleNamespace as NS
    from mtplx.attention_context import vision_rope, vision_rope_state
    from mtplx.frankie.completions import Completions

    service = Completions.__new__(Completions)
    service.engine = NS(mtp=2)
    service.active = {}
    service.mtp_steps = lambda job: iter(())
    job = NS(id="independent-request")
    with vision_rope("voice-image", 17):
        service.insert(job)
        assert vision_rope_state() == ("voice-image", 17)
        assert job.context.run(vision_rope_state) is None
