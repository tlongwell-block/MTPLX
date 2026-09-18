"""Low-reserve dispatch is restricted to opted-in disposable ear work."""

import asyncio
from threading import Event
from types import SimpleNamespace as NS

import pytest
from mtplx.frankie.completions import Completions, Job


def fixture(mtp):
    marker = object()
    model = NS(_mtplx_feature_stream=marker)
    engine = NS(mtp=mtp, runtime=NS(model=model), background_step=None)
    service = Completions(engine, None)

    def forbidden(*args, **kwargs):
        pytest.fail("Urgent preparation must not execute normal brain/HTTP work.")

    service.prepare = service.insert = service.step_mtp = service.warm_listener = forbidden
    return service, model, marker


def enqueue(service, callback=None, *, internal=True, prepare_only=True, urgent=True):
    job = Job({}, True, asyncio.get_running_loop())
    job.internal, job.prepare_only, job.urgent_prepare = internal, prepare_only, urgent
    job.prepare_callback = callback
    service.pending.append(job)
    service.jobs.add(job)
    return job


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("lead,run", [(0, False), (.119, False), (.12, True), (.4, True),
                                     (float("nan"), False)])
def test_short_ear_reserve_is_explicit_and_pending_http_is_untouched(mtp, lead, run):
    async def check():
        service, model, marker = fixture(mtp)
        calls = []

        def ear(job):
            assert model._mtplx_feature_stream is None
            calls.append(job)
            return "complete prefix transcript"

        urgent = enqueue(service, ear)
        http = enqueue(service, internal=False, prepare_only=False, urgent=False)
        assert service.urgent_step(lead) is run
        assert model._mtplx_feature_stream is marker
        assert calls == ([urgent] if run else [])
        assert list(service.pending) == ([http] if run else [urgent, http])
        assert service.jobs == ({http} if run else {urgent, http})
        assert not service.active and service.batch is None
        if run:
            event = urgent.events.get_nowait()
            assert event["prepared"] == "complete prefix transcript"
            assert event["usage"]["listener_owner_slices"] == 1
            assert urgent.done and urgent.prepare_callback is None

    asyncio.run(check())


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("head", ["none", "http", "brain", "final_ear"])
def test_no_opt_in_never_advances_any_work_even_with_active_http(mtp, head):
    async def check():
        service, model, marker = fixture(mtp)
        if head != "none":
            enqueue(service, lambda job: pytest.fail("Unmarked operation ran"),
                    internal=head != "http", prepare_only=head == "final_ear", urgent=False)
        active = Job({}, True, asyncio.get_running_loop())
        service.active["http"] = active
        service.jobs.add(active)
        pending, jobs = list(service.pending), set(service.jobs)
        assert service.urgent_step(.9) is False
        assert list(service.pending) == pending and service.jobs == jobs
        assert service.active == {"http": active}
        assert model._mtplx_feature_stream is marker

    asyncio.run(check())


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("outcome", ["cancel_before", "cancel_during", "error"])
def test_urgent_cancel_and_error_retire_without_falling_through(mtp, outcome):
    async def check():
        service, model, marker = fixture(mtp)
        calls = []

        def ear(job):
            calls.append(job)
            assert model._mtplx_feature_stream is None
            if outcome == "error":
                raise RuntimeError("Fixture ear error")
            job.cancelled.set()
            return "discarded"

        urgent = enqueue(service, ear)
        http = enqueue(service, internal=False, prepare_only=False, urgent=False)
        if outcome == "cancel_before":
            urgent.cancelled.set()
        assert service.urgent_step(.12) is True
        assert model._mtplx_feature_stream is marker
        assert calls == ([] if outcome == "cancel_before" else [urgent])
        assert service.jobs == {http} and list(service.pending) == [http]
        assert not urgent.done and urgent.prepare_callback is None
        if outcome == "error":
            assert urgent.events.get_nowait() == {"error": "Fixture ear error"}
        assert urgent.events.empty()

    asyncio.run(check())


@pytest.mark.parametrize("urgent", [False, True])
def test_submit_prepare_explicit_opt_in_and_callback_survives_new_admission(urgent):
    async def check():
        service, _, _ = fixture(2)
        first = service.submit_prepare(lambda job: "first", urgent=urgent)
        service.driver.cancel()
        await asyncio.gather(service.driver, return_exceptions=True)
        assert first.urgent_prepare is urgent
        if not urgent:
            assert getattr(service.engine, "urgent_background_step", None) is None
            return
        dispatch = service.engine.urgent_background_step
        assert dispatch(.12) is True
        assert dispatch(.12) is False
        second = service.submit_prepare(lambda job: "second", urgent=True)
        service.driver.cancel()
        await asyncio.gather(service.driver, return_exceptions=True)
        assert dispatch(.12) is True
        assert second.done and not service.jobs

    asyncio.run(check())


@pytest.mark.parametrize("kwargs", [{"urgent": True}, {"urgent": True, "internal": True},
                                    {"urgent": 1, "internal": True, "prepare_only": True,
                                     "prepare": lambda job: None}])
def test_urgent_cannot_be_attached_to_http_or_brain_jobs(kwargs):
    async def check():
        service, _, _ = fixture(2)
        with pytest.raises(ValueError):
            service.submit({}, True, **kwargs)
        assert not service.jobs and not service.pending and service.driver is None

    asyncio.run(check())


@pytest.mark.parametrize("mtp", [0, 2])
def test_engine_dispatches_between_audio_chunks_below_normal_reserve(monkeypatch, mtp):
    from mtplx.frankie import engine as module

    clock = NS(now=0.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module, "mx", NS(stack=lambda rows: rows))
    monkeypatch.setattr(module, "thinking_guard", lambda *args: None)
    values = {1: "Hello.", 2: " Welcome."}

    class Detokenizer:
        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = values[token]

        def finalize(self):
            self.last_segment = ""

    class Features:
        def __init__(self, runtime, offset, received):
            self.received, self.tokens, self.emitted = received, [], 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def commit(self, tokens):
            self.tokens.extend(tokens)
            self.emitted = len(self.tokens)
            self.received(tokens, [None] * len(tokens))

        def flush(self, **kwargs):
            pass

    def generate(runtime, ids, *, token_callback, **kwargs):
        token_callback([1, 2])
        return NS(stats=NS(to_dict=dict), finish_reason="stop")

    def speak(text, rows):
        for _ in range(2):
            yield [0.0] * 3840
            clock.now += .16

    monkeypatch.setattr(module, "CommittedFeatures", Features)
    monkeypatch.setattr(module, "generate_mtpk", generate)
    monkeypatch.setattr(module, "generate_ar", generate)
    engine = module.Frankie.__new__(module.Frankie)
    engine.mtp, engine.runtime, engine.bank = mtp, NS(model=NS()), None
    engine.audio = NS(speak=speak, reset_speech_context=lambda: None)
    engine.tokenizer = NS(detokenizer=Detokenizer(), eos_token_ids=[0],
                          decode=lambda tokens: "".join(values[token] for token in tokens))
    engine.prompt = lambda *args, **kwargs: ([0], None)
    urgent_calls, normal_calls, events = [], [], []

    def urgent(lead):
        if lead < .12 or urgent_calls:
            return False
        urgent_calls.append((lead, len([kind for kind, _ in events if kind == "audio"])))
        clock.now += .01
        return True

    engine.urgent_background_step = urgent
    engine.background_step = lambda lead: normal_calls.append(lead) or False
    result = engine.respond([], {"output_modalities": ["audio"], "max_output_tokens": 4,
                                 "thinking": "off"}, lambda *event: events.append(event),
                            Event(), session_id="urgent-fixture")
    assert urgent_calls == [(.16, 1)] and normal_calls == []
    assert result["text"] == "Hello. Welcome." and result["audio_seconds"] == .64
    assert len([kind for kind, _ in events if kind == "audio"]) == 4
