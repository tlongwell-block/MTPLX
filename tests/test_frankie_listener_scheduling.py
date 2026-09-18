"""Bounded same-owner listener slices and internal-only accounting, no models."""

import asyncio
from collections import deque
import contextvars
from threading import Event
from types import SimpleNamespace as NS

import pytest

from mtplx.frankie import completions as c


def service_with_steps(*, internal):
    seen = []
    job = NS(internal=internal, cancelled=Event(), context=contextvars.Context(), ids=[1])
    def steps():
        while True:
            seen.append(job.prefill_step_size)
            yield None
    job.steps = steps()
    service = c.Completions.__new__(c.Completions)
    service.active, service.pending = {"job": job}, deque()
    return service, job, seen


@pytest.mark.parametrize("setting,expected", [(None, 64), ("32", 32), ("64", 64),
                                              ("128", 128), ("999", 128), ("1", 32), ("invalid", 64)])
def test_internal_budget_is_bounded_and_http_budget_unchanged(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv("FRANKIE_LISTENER_PREFILL_TOKENS", raising=False)
    else:
        monkeypatch.setenv("FRANKIE_LISTENER_PREFILL_TOKENS", setting)
    for internal in (False, True):
        service, job, seen = service_with_steps(internal=internal)
        for voice in (False, True, False):
            service.step_mtp(voice)
        assert seen == ([expected] * 3 if internal else [64, 32, 64])
        if internal:
            assert job.owner_slices == 3
        else:
            assert c.Completions.owner_usage(job) == {}
            assert not hasattr(job, "owner_work_ms")
        job.steps.close()


def test_owner_accounting_includes_final_stop_iteration_before_usage(monkeypatch):
    ticks = iter((1.0, 1.025))
    monkeypatch.setattr(c.time, "perf_counter", lambda: next(ticks))
    service, job, _ = service_with_steps(internal=True)
    job.steps.close()
    def done():
        if False:
            yield
        return "stop"
    job.steps = done()
    results = []
    service.close_mtp_job = lambda job: None
    service.finish = lambda job, reason: results.append((reason, service.owner_usage(job)))
    service.step_mtp(True)
    assert results == [("stop", {"listener_owner_work_ms": 25.0,
                                  "listener_owner_slices": 1, "listener_owner_max_slice_ms": 25.0})]


def test_failed_owner_slice_is_accounted_without_swallowing_exception(monkeypatch):
    ticks = iter((1.0, 1.02, 2.0, 2.05))
    monkeypatch.setattr(c.time, "perf_counter", lambda: next(ticks))
    job = NS(internal=True)
    assert c.Completions.owner_work(job, lambda: "ok") == "ok"
    def fail():
        raise RuntimeError("Fixture failure")
    with pytest.raises(RuntimeError, match="Fixture failure"):
        c.Completions.owner_work(job, fail)
    assert c.Completions.owner_usage(job) == {"listener_owner_work_ms": 70.0,
                                            "listener_owner_slices": 2, "listener_owner_max_slice_ms": 50.0}


def test_voice_reserve_remains_six_hundred_ms():
    service = c.Completions.__new__(c.Completions)
    service.engine = NS(mtp=2)
    service.pending, service.active = deque(), {}
    steps = []
    service.step = lambda **kwargs: steps.append(kwargs)
    assert service.voice_step(.599) is False and steps == []
    assert service.voice_step(.6) is True and steps == [{"voice": True}]


@pytest.mark.parametrize("internal,chunk_size", [(False, 64), (True, 128)])
def test_only_internal_prefill_can_grow_past_existing_http_chunk_cap(monkeypatch, internal, chunk_size):
    from mtplx import generation
    seen = []
    def prepare(*args, **kwargs):
        seen.append(kwargs)
        yield "prepared"
    monkeypatch.setattr(generation.restore_or_prefill_prompt_state, "steps", prepare)
    service = c.Completions.__new__(c.Completions)
    service.engine = NS(runtime=object(), bank=object())
    job = NS(internal=internal, splice=None, ids=[1], id="job", cancelled=Event(), prefill_step_size=128)
    steps = service.mtp_steps(job)
    assert next(steps) == "prepared"
    assert seen[0]["prefill_chunk_size"] == chunk_size
    assert seen[0]["prefill_step_size"]() == 128
    steps.close()


def test_prepare_only_reports_internal_owner_time_without_brain_work(monkeypatch):
    ticks = iter((1.0, 1.012))
    monkeypatch.setattr(c.time, "perf_counter", lambda: next(ticks))
    async def check():
        job = c.Job({}, True, asyncio.get_running_loop())
        job.internal = job.prepare_only = True
        job.prepare_callback = lambda job: "actual-ear-evidence"
        service = c.Completions.__new__(c.Completions)
        service.jobs = {job}
        service.run_preparation(job)
        event = await job.receive()
        assert event["prepared"] == "actual-ear-evidence"
        assert event["usage"]["listener_owner_work_ms"] == 12.0
        assert event["usage"]["listener_owner_slices"] == 1
        assert not service.jobs
    asyncio.run(check())


def test_bpe_template_reuses_vocab_but_isolates_all_request_buffers():
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer
    class Tokenizer:
        clean_up_tokenization_spaces = False
        vocab = {"H": 0, "i": 1, "B": 2, "y": 3, "e": 4}
        constructions = 0
        @property
        def detokenizer(self):
            self.constructions += 1
            return BPEStreamingDetokenizer(self)
    tokenizer = Tokenizer()
    service = c.Completions.__new__(c.Completions)
    service.engine = NS(tokenizer=tokenizer)
    first, second = service.new_detokenizer(), service.new_detokenizer()
    for token in (0, 1):
        first.add_token(token)
    for token in (2, 3, 4):
        second.add_token(token)
    first.finalize()
    second.finalize()
    assert first.text == "Hi" and second.text == "Bye"
    assert first.tokens == [0, 1] and second.tokens == [2, 3, 4]
    assert first.tokenmap is second.tokenmap is service.detokenizer_template.tokenmap
    third = service.new_detokenizer()
    assert tokenizer.constructions == 1
    assert third.tokens == [] and third.text == "" and third._unflushed == "" and third.offset == 0
    assert service.detokenizer_template.tokens == [] and service.detokenizer_template.text == ""


def test_unknown_detokenizer_keeps_previous_fresh_construction_behavior():
    class Tokenizer:
        constructions = 0
        @property
        def detokenizer(self):
            self.constructions += 1
            return NS(reset=lambda: None)
    tokenizer = Tokenizer()
    service = c.Completions.__new__(c.Completions)
    service.engine = NS(tokenizer=tokenizer)
    assert service.new_detokenizer() is not service.new_detokenizer()
    assert tokenizer.constructions == 2 and getattr(service, "detokenizer_template", None) is None


def test_owner_warm_primes_bpe_template_once_before_first_live_request():
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer
    class Tokenizer:
        clean_up_tokenization_spaces = False
        vocab = {"H": 0, "i": 1}
        constructions = 0
        @property
        def detokenizer(self):
            self.constructions += 1
            return BPEStreamingDetokenizer(self)
    service = c.Completions.__new__(c.Completions)
    tokenizer = Tokenizer()
    service.engine = NS(mtp=0, tokenizer=tokenizer)
    service.listener_bank, service.listener_prefixes = None, set()
    service.warm_listener()
    service.warm_listener()
    request = service.new_detokenizer()
    assert tokenizer.constructions == 1
    assert request.tokenmap is service.detokenizer_template.tokenmap
    assert request.tokens == [] and request.text == ""


def test_owner_warm_does_not_repeatedly_prime_uncacheable_detokenizer():
    class Tokenizer:
        constructions = 0
        @property
        def detokenizer(self):
            self.constructions += 1
            return NS(reset=lambda: None)
    service = c.Completions.__new__(c.Completions)
    tokenizer = Tokenizer()
    service.engine = NS(mtp=0, tokenizer=tokenizer)
    service.listener_bank, service.listener_prefixes = None, set()
    service.warm_listener()
    service.warm_listener()
    assert tokenizer.constructions == 1
    service.new_detokenizer()
    assert tokenizer.constructions == 2


@pytest.mark.parametrize("costs,expected", [([.01, .01, .01, .01], 3), ([.01, .343, .01], 2), ([.3, .01], 1)])
def test_internal_burst_rechecks_declining_audio_lead_and_caps_three(monkeypatch, costs, expected):
    clock, calls = [0.0], []
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    service = c.Completions.__new__(c.Completions)
    service.engine = NS(mtp=2)
    service.pending = deque()
    service.active = {"listener": NS(internal=True, cancelled=Event())}
    def step(**kwargs):
        calls.append(kwargs)
        clock[0] += costs[len(calls) - 1]
    service.step = step
    assert service.voice_step(.8) is True
    assert len(calls) == expected
    assert calls[0] == {"voice": True}
    assert calls[1:] == [{"voice": True, "internal_only": True}] * (expected - 1)


@pytest.mark.parametrize("mtp,internal", [(2, False), (0, True)])
def test_http_and_non_mtp_batching_keep_exactly_one_step(monkeypatch, mtp, internal):
    monkeypatch.setattr(c.time, "monotonic", lambda: 0)
    service = c.Completions.__new__(c.Completions)
    service.engine = NS(mtp=mtp)
    service.pending = deque()
    service.active = {"job": NS(internal=internal, cancelled=Event())}
    calls = []
    service.step = lambda **kwargs: calls.append(kwargs)
    assert service.voice_step(10) is True
    assert calls == [{"voice": True}]


def test_pending_http_keeps_single_step_even_with_an_internal_job_active():
    service = c.Completions.__new__(c.Completions)
    service.engine = NS(mtp=2)
    service.pending = deque([NS(internal=False)])
    service.active = {"listener": NS(internal=True, cancelled=Event())}
    calls = []
    service.step = lambda **kwargs: calls.append(kwargs)
    service.voice_step(10)
    assert calls == [{"voice": True}]


def test_internal_burst_preserves_admission_and_voice_features_during_http_enqueue_race(monkeypatch):
    monkeypatch.setattr(c.time, "monotonic", lambda: 0)
    class Work:
        def __init__(self, internal, uid):
            self.internal, self.uid = internal, uid
            self.cancelled = Event()
            self.context = contextvars.Context()
            self.ids = [1]
    listener, http = Work(True, "listener"), Work(False, "http")
    voice_features = object()
    model = NS(_mtplx_feature_stream=voice_features)
    service = c.Completions.__new__(c.Completions)
    service.engine = NS(mtp=2, runtime=NS(model=model))
    service.pending, service.active, service.jobs = deque([listener]), {}, {listener, http}
    actions = []
    def prepare(job):
        assert job is listener and model._mtplx_feature_stream is None
        actions.append("prepare")
    def steps():
        while True:
            assert model._mtplx_feature_stream is None
            actions.append("decode")
            yield None
    def insert(job):
        job.steps = steps()
        service.active[job.uid] = job
    service.prepare, service.insert = prepare, insert
    original_step = service.step
    def raced_step(**kwargs):
        if kwargs.get("internal_only"):
            # The event-loop thread may enqueue HTTP after the burst check.
            service.pending.appendleft(http)
        original_step(**kwargs)
        assert model._mtplx_feature_stream is voice_features
    service.step = raced_step
    assert service.voice_step(.8) is True
    assert actions == ["prepare", "decode"]
    assert list(service.pending) == [http]
    assert service.active == {"listener": listener} and service.jobs == {listener, http}
    listener.steps.close()
