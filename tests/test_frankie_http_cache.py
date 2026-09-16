"""HTTP MTP shares the prompt bank without sharing mutable request caches."""

import contextlib
import contextvars
import os
from threading import Event
from types import SimpleNamespace as NS

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache
import pytest

from mtplx import generation as g
from mtplx.cache_state import snapshot_cache
from mtplx.frankie.completions import Completions
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank
from test_generation_sustained import AcceptingTinyMTPModel, _runtime
from test_tail_ar_warm_restore_identity import HistoryCountModel, VOCAB


class CachedMTPModel(HistoryCountModel, AcceptingTinyMTPModel):
    """Real mutable KV state; target logits depend on the whole prefix."""

    def __init__(self):
        super().__init__()
        self.mtp = NS(_mtplx_lora_targets=[])
        self.model = NS(embed_tokens=lambda ids: mx.eye(VOCAB)[ids])

    def __call__(self, input_ids, *, input_embeddings=None, **kwargs):
        if input_embeddings is not None:
            input_ids = mx.argmax(input_embeddings, axis=-1)
        return super().__call__(input_ids, **kwargs)

    def make_mtp_cache(self):
        return [KVCache()]

    def mtp_update_cache(self, hidden_states, next_token_ids, *, mtp_cache, **kwargs):
        rows = next_token_ids.astype(mx.float32)[:, None, :, None]
        mtp_cache[0].update_and_fetch(rows, rows)
        return hidden_states

    def mtp_forward(self, hidden_states, next_token_ids, *, mtp_cache,
                    return_hidden=False, **kwargs):
        self.mtp_update_cache(hidden_states, next_token_ids, mtp_cache=mtp_cache)
        logits = mx.zeros((1, next_token_ids.shape[1], VOCAB))
        return (logits, hidden_states) if return_hidden else logits


@pytest.fixture(autouse=True)
def cpu_and_eager(monkeypatch):
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_SESSION_MIN_REUSE", "1")
    try:
        yield
    finally:
        mx.set_default_device(device)


def bank():
    return SessionBank(max_entries=6, max_bytes=1 << 24, per_session_max_bytes=1 << 24)


def drain(steps, context=None):
    context = context or contextvars.Context()
    progress = []
    while True:
        try:
            progress.append(context.run(next, steps))
        except StopIteration as done:
            return done.value, progress


def prepare(runtime, ids, cache, *, budget=lambda: 64, **kwargs):
    return g.restore_or_prefill_prompt_state.steps(
        runtime, ids, mtp_history_policy="committed", session_bank=cache,
        restore_mode="clone", session_id="prepare", store_prefix_snapshot=False,
        prefill_chunk_size=64, prefill_step_size=budget, **kwargs,
    )


def generate(runtime, ids, cache, *, seed=5, depth=2, temperature=0.7):
    state, _ = drain(prepare(runtime, ids, cache))
    result = g.generate_mtpk(
        runtime, ids, _prompt_state=state, session_bank=cache, session_id=str(seed),
        commit_prompt_state_to_bank=True, mtp_history_policy="committed",
        speculative_depth=depth, max_tokens=6, seed=seed, stop_token_ids=set(),
        sampler=SamplerConfig(temperature=temperature, top_p=1, top_k=0),
    )
    return result, state


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("temperature", [0, 0.7])
def test_repeat_and_extended_prefix_match_cold_and_clone_caches(depth, temperature):
    runtime = _runtime(CachedMTPModel())
    cache = bank()
    prompt = [3, 1, 4, 1, 5, 9] * 20
    random_before = [x.tolist() for x in mx.random.state]
    first, _ = generate(runtime, prompt, cache, depth=depth, temperature=temperature)
    runtime.model.calls.clear()
    warm_state, progress = drain(prepare(runtime, prompt, cache))
    assert warm_state.cached_tokens == len(prompt)
    assert runtime.model.calls == []
    assert progress == []
    entry = cache._entries[tuple(prompt)]
    assert entry.cache_ref is None
    assert entry.mtp_history_cache_ref is None
    a, _ = drain(prepare(runtime, prompt, cache))
    b, _ = drain(prepare(runtime, prompt, cache))
    assert a.trunk_cache[0] is not b.trunk_cache[0]
    assert a.committed_mtp_cache[0] is not b.committed_mtp_cache[0]
    repeat, _ = generate(runtime, prompt, cache, depth=depth, temperature=temperature)
    assert repeat.tokens == first.tokens
    extended = prompt + [2, 6, 5, 3] * 25
    warm, state = generate(runtime, extended, cache, depth=depth, temperature=temperature)
    cold, _ = generate(runtime, extended, None, depth=depth, temperature=temperature)
    assert state.cached_tokens >= len(prompt)
    assert warm.tokens == cold.tokens
    assert [x.tolist() for x in mx.random.state] == random_before


def test_warm_suffix_budget_adapts_and_fused_shortcut_is_bounded():
    runtime = _runtime(CachedMTPModel())
    cache = bank()
    prefix = [3, 1, 4] * 30
    generate(runtime, prefix, cache)
    budget = 64
    runtime.model.calls.clear()
    steps = prepare(runtime, prefix + [2] * 300, cache, budget=lambda: budget)
    context = contextvars.Context()
    assert context.run(next, steps)["prefill_tokens"] == len(prefix) + 64
    budget = 32
    assert context.run(next, steps)["prefill_tokens"] == len(prefix) + 96
    state, _ = drain(steps, context)
    assert state.cached_tokens == len(prefix)
    assert max(runtime.model.calls) <= 64
    runtime.model.calls.clear()
    state, _ = drain(prepare(runtime, prefix + [2] * 50, cache, budget=lambda: 32))
    assert state.cached_tokens == len(prefix)
    assert max(runtime.model.calls) <= 32


@pytest.mark.parametrize("warm", [False, True])
def test_prefill_close_restores_scopes_without_publishing_partial_state(monkeypatch, warm):
    runtime = _runtime(CachedMTPModel())
    cache = bank()
    prefix = [3, 1, 4] * 30
    if warm:
        generate(runtime, prefix, cache)
    before = tuple(cache._entries)
    scope = contextvars.ContextVar("http-test-scope", default=())

    def named_scope(name):
        @contextlib.contextmanager
        def enter(*args, **kwargs):
            token = scope.set(scope.get() + (name,))
            try:
                yield
            finally:
                scope.reset(token)
        return enter

    monkeypatch.setattr(g, "_vision_rope_scope_for", named_scope("vision"))
    monkeypatch.setattr(g, "_ple_first_gather_early_scope", named_scope("ple"))
    context = contextvars.Context()
    steps = prepare(runtime, prefix + [2] * 300, cache)
    context.run(next, steps)
    assert context.run(scope.get) == ("ple", "vision")
    assert scope.get() == ()
    context.run(steps.close)
    assert context.run(scope.get) == ()
    assert tuple(cache._entries) == before


def test_http_iterator_uses_shared_bank_once_and_reports_hits(monkeypatch):
    runtime = _runtime(CachedMTPModel())
    cache = bank()
    service = Completions.__new__(Completions)
    service.engine = NS(runtime=runtime, bank=cache, mtp=2,
                        tokenizer=NS(eos_token_ids=set()))
    service.emit_text = lambda *args: None

    class Detokenizer:
        last_segment = ""

        def add_token(self, token):
            pass

    puts = []
    original_put = cache.put

    def put(**kwargs):
        puts.append(kwargs)
        return original_put(**kwargs)

    monkeypatch.setattr(cache, "put", put)
    # Exercise the normally eager large-prefix store gate without a large test.
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1")
    outputs = []
    for i in range(2):
        job = NS(ids=[3, 1, 4] * 30, id=f"http-{i}", splice=None,
                 prefill_step_size=64, cancelled=Event(), tokens=[],
                 detokenizer=Detokenizer(), data={"thinking": "off", "max_tokens": 6,
                 "seed": 5, "temperature": 0.7, "top_p": 1, "top_k": 0})
        drain(service.mtp_steps(job))
        outputs.append(job.tokens)
        assert job.cached_tokens == (len(job.ids) if i else 0)
    assert outputs[0] == outputs[1]
    assert len(puts) == 1
    assert puts[0]["keep_live_ref"] is False


def test_interleaved_branches_and_cancelled_suffix_keep_banked_prefix_immutable():
    runtime = _runtime(CachedMTPModel())
    cache = bank()
    prefix = [3, 1, 4] * 30
    generate(runtime, prefix, cache)
    branches = [prefix + [2] * 90, prefix + [6] * 100]
    expected = [generate(runtime, ids, None, seed=i)[0].tokens
                for i, ids in enumerate(branches)]
    contexts = [contextvars.Context(), contextvars.Context()]

    def request(ids, seed):
        state = yield from prepare(runtime, ids, cache, budget=lambda: 32)
        return (yield from g.generate_mtpk.steps(runtime, ids, _prompt_state=state,
            mtp_history_policy="committed", speculative_depth=2, max_tokens=6,
            sampler=SamplerConfig(temperature=0.7, top_p=1, top_k=0),
            seed=seed, stop_token_ids=set()))

    streams = [request(ids, i) for i, ids in enumerate(branches)]
    results = [None, None]
    while any(value is None for value in results):
        for i, stream in enumerate(streams):
            if results[i] is None:
                try:
                    contexts[i].run(next, stream)
                except StopIteration as done:
                    results[i] = done.value.tokens
    assert results == expected
    cancelled = Event()
    steps = prepare(runtime, prefix + [7] * 400, cache,
                    abort_check=cancelled.is_set)
    context = contextvars.Context()
    context.run(next, steps)
    cancelled.set()
    with pytest.raises(g.PostcommitAbort):
        context.run(next, steps)
    warm, state = generate(runtime, prefix, cache)
    cold, _ = generate(runtime, prefix, None)
    assert state.cached_tokens == len(prefix)
    assert warm.tokens == cold.tokens


def test_image_digest_and_rope_context_survive_cooperative_cache_restore():
    from mtplx.attention_context import vision_rope_state
    from mtplx.vision.splice import VisionSplice

    runtime = _runtime(CachedMTPModel())
    cache = bank()
    ids = [3] * 70 + [31, 31] + [1] * 60

    def request(digest, image_token):
        splice = VisionSplice(image_pad_token_id=31,
            embeddings=mx.eye(VOCAB)[mx.array([image_token, image_token])],
            image_digests=(digest,), pad_counts=(2,), mrope_delta=digest)
        context = contextvars.Context()
        steps = prepare(runtime, ids, cache, vision_splice=splice)
        while True:
            try:
                context.run(next, steps)
                assert context.run(vision_rope_state) == (None, digest)
                assert vision_rope_state() is None
            except StopIteration as done:
                state = done.value
                break
        assert context.run(vision_rope_state) is None
        result = g.generate_mtpk(runtime, ids, _prompt_state=state,
            vision_splice=splice, session_bank=cache, session_id=str(digest),
            commit_prompt_state_to_bank=True, mtp_history_policy="committed",
            max_tokens=4, speculative_depth=2, stop_token_ids=set(),
            sampler=SamplerConfig(temperature=0), seed=5)
        return result, state

    a, cold_a = request(11, 2)
    a_repeat, warm_a = request(11, 2)
    b, cold_b = request(22, 7)
    b_repeat, warm_b = request(22, 7)
    assert cold_a.cached_tokens == 0
    assert warm_a.cached_tokens == len(ids)
    assert cold_b.cached_tokens <= 70
    assert warm_b.cached_tokens == len(ids)
    assert a.tokens == a_repeat.tokens
    assert b.tokens == b_repeat.tokens
    # Both requests may share an argmax, but the model must see different rows.
    assert not mx.array_equal(cold_a.logits, cold_b.logits).item()
    assert mx.array_equal(cold_b.logits, warm_b.logits).item()


def test_scheduler_restores_global_layout_hint_even_when_request_fails(monkeypatch):
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "777")
    service = Completions.__new__(Completions)
    job = NS(ids=[0] * 123, cancelled=Event(), context=contextvars.Context(),
             uid="job", emit=lambda event: None)

    def steps():
        assert os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] == "123"
        yield
        raise ValueError("synthetic failure")

    job.steps = steps()
    service.active = {"job": job}
    service.pending = []
    service.jobs = {job.uid}
    service.close_mtp_job = lambda _: service.active.clear()
    service.step_mtp(False)
    assert os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] == "777"
    service.step_mtp(False)
    assert os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] == "777"


def test_divergent_history_restores_true_recurrent_boundary():
    class HybridModel(CachedMTPModel):
        def make_cache(self):
            recurrent = ArraysCache(1)
            recurrent[0] = mx.zeros((1, 1), mx.int32)
            return [KVCache(), recurrent]

        def __call__(self, input_ids, *, cache, **kwargs):
            cache[1][0] = cache[1][0] + mx.sum(input_ids)
            return super().__call__(input_ids, cache=cache, **kwargs)

    runtime = _runtime(HybridModel())
    cache = bank()
    original = [3, 1, 4] * 300
    state, _ = drain(prepare(runtime, original, cache))
    assert state.gdn_boundaries
    cache.put(runtime=runtime, token_ids=original, cache=state.trunk_cache,
        logits=state.logits, hidden=state.hidden, hidden_variant="post_norm",
        session_id="source", mtp_history_policy="committed",
        mtp_history_snapshot=snapshot_cache(state.committed_mtp_cache),
        snapshot_epoch=len(original), mtp_snapshot_epoch=len(original),
        gdn_boundaries=state.gdn_boundaries)
    # The changed tail stands in for a different tool result / call id.
    changed = original[:600] + [7] * 300
    warm, progress = drain(prepare(runtime, changed, cache, budget=lambda: 32))
    cold, _ = drain(prepare(runtime, changed, None, budget=lambda: 32))
    assert 0 < warm.cached_tokens <= 600
    assert progress
    assert mx.array_equal(warm.logits, cold.logits).item()
    assert mx.array_equal(warm.trunk_cache[1][0], cold.trunk_cache[1][0]).item()
    assert warm.trunk_cache[1][0].item() == sum(changed)
    assert mx.array_equal(warm.committed_mtp_cache[0].state[0],
                          cold.committed_mtp_cache[0].state[0]).item()
