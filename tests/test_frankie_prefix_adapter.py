"""Real owner scheduler, fake ear/brain: no model loads or GPU operations."""
import asyncio
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip("mlx.core")  # Completions.step imports MLX, but these tests run no tensor math.
from mtplx.frankie.completions import Completions
from mtplx.frankie.listening import StreamingListener
from mtplx.frankie.perception import RealtimeListener


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.001)


def snapshot(ms=400, *, final=False, rate=24000):
    core = StreamingListener()
    core.begin("response", "utterance", sample_rate=rate, start_ms=100)
    core.append(np.full(ms * rate // 1000, 1234, dtype="<i2").tobytes())
    if final:
        core.finish_input()
    return core.request()


class Ear:
    def __init__(self, model, texts):
        self.model, self.texts = model, iter(texts)
        self.calls = []
        self.hook = None

    def hear(self, pcm, rate):
        assert self.model._mtplx_feature_stream is None
        self.calls.append((pcm.copy(), rate, threading.get_ident()))
        if self.hook:
            self.hook()
        return object(), next(self.texts)


class FakeBrain(Completions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts = []
        self.action = "wait"

    def prepare(self, job):
        job.prepare_callback(job)
        self.prompts.append(copy.deepcopy(job.data))
        job.emit({"listener_prepare_seconds": .012})

    def insert(self, job):
        # Only Qwen's computation is stubbed, not owner scheduling or admission.
        job.done = True
        job.emit({"finish_reason": "stop", "message": {"content": self.action},
                  "usage": {"completion_tokens": 2}})
        self.jobs.discard(job)


async def fixture(check, texts=(), *, mtp=2, real_service=False):
    marker = object()
    model = NS(_mtplx_feature_stream=marker)
    ear = Ear(model, texts)
    engine = NS(runtime=NS(model=model), audio=ear, mtp=mtp, background_step=None)
    with ThreadPoolExecutor(max_workers=1) as owner:
        service = (Completions if real_service else FakeBrain)(engine, owner, slots=1)
        session = NS(engine=engine, closed=False, items=[], settings={"assistant_name": "Frankie"},
                     clock_ms=999, task_ledger=NS(tasks={}))
        listener = RealtimeListener(session, service)
        try:
            await check(listener, service, session, ear, owner)
        finally:
            listener.cancel()
            if service.driver is not None:
                await asyncio.wait_for(service.driver, 2)
    assert model._mtplx_feature_stream is marker


def test_prefix_uses_complete_current_ctc_on_actual_single_owner_and_no_history_cache():
    async def check(listener, service, session, ear, owner):
        session.items = [{"role": "user", "content": [{"_pcm": [1, 2], "type": "input_audio"}]}]
        before = copy.deepcopy(session.items)
        snap = snapshot()
        decision, evidence, stats = await listener.classify_prefix(
            snap, heard_text="Keep explaining the route.", pending_work=True, assistant_name="Frankie")
        assert decision.action == "wait" and decision.confidence is None
        assert not hasattr(decision, "addressed_to_assistant")
        assert not hasattr(decision, "sufficient_evidence")
        assert evidence.snapshot is snap and evidence.text == "Stop worrying."
        assert evidence.previous_text == "Stop"
        assert evidence.previous_samples == 240 * 24
        assert not evidence.stable(160)
        payload = json.loads(service.prompts[0]["messages"][-1]["content"])
        assert payload["user_prefix"] == "Stop worrying."  # Never the LCP "stop".
        assert payload["assistant_already_heard"] == "Keep explaining the route."
        assert payload["pending_work"] and payload["assistant_name"] == "Frankie"
        assert payload["user_speaking"] and not payload["prefix_final"]
        assert not payload["prefix_stable"]
        assert decision.observed_ms == 500 and decision.revision == snap.revision
        assert [len(call[0]) for call in ear.calls] == [240 * 24, 400 * 24]
        assert len({call[2] for call in ear.calls}) == 1
        assert ear.calls[0][2] != threading.get_ident()
        assert session.items == before and not service.jobs and listener.job is None
        assert stats["ear_ms"] == 12 and stats["completion_tokens"] == 2
    asyncio.run(fixture(check, ["Stop", "Stop worrying."]))


@pytest.mark.parametrize("rate", [16000, 24000])
def test_short_final_prefix_has_no_fabricated_earlier_audio(rate):
    async def check(listener, service, session, ear, owner):
        decision, evidence, _ = await listener.classify_prefix(
            snapshot(120, final=True, rate=rate), heard_text="Hello.", pending_work=False,
            assistant_name=None)
        assert len(ear.calls) == 1 and ear.calls[0][1] == rate
        assert evidence.previous_text is None and evidence.previous_samples is None
        payload = json.loads(service.prompts[0]["messages"][-1]["content"])
        assert payload["prefix_final"] and not payload["user_speaking"]
        assert not payload["prefix_stable"]
        assert decision.observed_ms == 220
    asyncio.run(fixture(check, ["Yes."]))


def test_agreeing_complete_snapshots_are_marked_lexically_stable_only():
    async def check(listener, service, session, ear, owner):
        _, evidence, _ = await listener.classify_prefix(snapshot(), heard_text="", pending_work=False,
                                                       assistant_name="Frankie")
        assert evidence.stable(160)
        assert json.loads(service.prompts[0]["messages"][-1]["content"])["prefix_stable"]
    asyncio.run(fixture(check, ["No.", "No."]))


@pytest.mark.parametrize("mtp", [0, 2])
def test_recheck_is_ear_only_no_brain_warm_prompt_tokenizer_or_cache(mtp):
    async def check(listener, service, session, ear, owner):
        def forbidden(*args, **kwargs):
            pytest.fail("A prepare-only ear recheck must not touch brain preparation.")
        service.prepare = service.warm_listener = service.insert = forbidden
        snap = snapshot()
        evidence = await listener.recheck_prefix(snap)
        assert evidence.snapshot is snap and evidence.text == "Don't stop."
        assert evidence.previous_text is None and len(ear.calls) == 1
        assert ear.calls[0][2] != threading.get_ident()
        assert not service.jobs and not service.pending and not service.active
        assert service.listener_bank is None and service.batch is None
        assert listener.job is None
    asyncio.run(fixture(check, ["Don't stop."], mtp=mtp, real_service=True))


def test_preparation_uses_reserved_internal_admission_and_priority():
    async def check(listener, service, session, ear, owner):
        blocked = threading.Event()
        owner.submit(blocked.wait)
        try:
            normal = service.submit({}, True, prepare=lambda job: None)
            observed = []
            prepared = service.submit_prepare(lambda job: observed.append("ear") or "ready")
            assert service.pending[0] is prepared
            with pytest.raises(OverflowError, match="listener"):
                service.submit_prepare(lambda job: None)
            with pytest.raises(OverflowError, match="HTTP"):
                service.submit({}, True)
            with pytest.raises(ValueError, match="internal"):
                service.submit({}, True, prepare=lambda job: None, prepare_only=True)
            blocked.set()
            result = await asyncio.wait_for(prepared.receive(), 2)
            assert result["prepared"] == "ready" and observed == ["ear"]
            await until(lambda: not service.jobs)
            assert normal.done
        finally:
            blocked.set()
    asyncio.run(fixture(check))


def test_cancelled_queued_prefix_keeps_owner_reservation_until_actual_retirement():
    async def check(listener, service, session, ear, owner):
        blocked = threading.Event()
        owner.submit(blocked.wait)
        try:
            task = asyncio.create_task(listener.classify_prefix(snapshot(), heard_text="", pending_work=False,
                                                               assistant_name="Frankie"))
            await until(lambda: listener.job is not None)
            job = listener.job
            listener.cancel()
            await asyncio.sleep(.025)
            assert not task.done() and job in service.jobs and listener.job is job
            assert not ear.calls
            blocked.set()
            with pytest.raises(RuntimeError, match="consumer"):
                await asyncio.wait_for(task, 2)
            assert job not in service.jobs and listener.job is None
        finally:
            blocked.set()
    asyncio.run(fixture(check))


def test_recheck_supersedes_prefix_only_after_old_owner_returns():
    async def check(listener, service, session, ear, owner):
        blocked, entered = threading.Event(), threading.Event()
        def hold_first():
            if len(ear.calls) == 1:
                entered.set()
                assert blocked.wait(2)
        ear.hook = hold_first
        first = asyncio.create_task(listener.classify_prefix(snapshot(), heard_text="", pending_work=False,
                                                             assistant_name="Frankie"))
        try:
            await until(entered.is_set)
            old = listener.job
            second = asyncio.create_task(listener.recheck_prefix(snapshot(720)))
            await until(old.cancelled.is_set)
            assert len(service.jobs) == 1 and old in service.jobs
            blocked.set()
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(first, 2)
            evidence = await asyncio.wait_for(second, 2)
            assert evidence.text == "No worries."
            assert [len(row[0]) for row in ear.calls] == [240 * 24, 720 * 24]
            assert not service.jobs and listener.job is None
        finally:
            blocked.set()
    asyncio.run(fixture(check, ["No", "No worries."]))


def test_recheck_failure_retires_and_restores_feature_capture():
    async def check(listener, service, session, ear, owner):
        def fail():
            raise ValueError("Synthetic ear failure")
        ear.hook = fail
        with pytest.raises(RuntimeError, match="Synthetic ear failure"):
            await listener.recheck_prefix(snapshot())
        assert not service.jobs and listener.job is None
        ear.hook = None
        evidence = await listener.recheck_prefix(snapshot())
        assert evidence.text == "Recovered."
    asyncio.run(fixture(check, ["Recovered."], real_service=True))


def test_closed_session_does_not_submit_any_owner_work():
    async def check(listener, service, session, ear, owner):
        session.closed = True
        with pytest.raises(RuntimeError, match="cancelled"):
            await listener.recheck_prefix(snapshot())
        assert not ear.calls and not service.jobs
    asyncio.run(fixture(check))


def test_task_cancellation_waits_for_real_owner_and_next_final_work_can_start():
    async def check(listener, service, session, ear, owner):
        blocked = threading.Event()
        owner.submit(blocked.wait)
        task = asyncio.create_task(listener.recheck_prefix(snapshot()))
        try:
            await until(lambda: listener.job is not None)
            old = listener.job
            task.cancel()
            await asyncio.sleep(.02)
            assert old.cancelled.is_set() and old in service.jobs and not task.done()
            blocked.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            evidence = await listener.recheck_prefix(snapshot(120, final=True))
            assert evidence.snapshot.final and evidence.text == "Final input."
            assert not service.jobs and listener.job is None
        finally:
            blocked.set()
    asyncio.run(fixture(check, ["Final input."], real_service=True))


@pytest.mark.parametrize("mode", ["observe", "semantic"])
def test_v5_final_nine_second_input_is_complete_and_never_reuses_prefix_features(mode):
    async def check(listener, service, session, ear, owner):
        session.settings["streaming_listener"] = mode
        _, prefix, _ = await listener.classify_prefix(snapshot(), heard_text="", pending_work=False,
                                                      assistant_name="Frankie")
        assert prefix.text == "No"
        pcm = np.linspace(-.2, .2, 9 * 24000, dtype=np.float32)
        item = {"id": "final-input", "content": [{"type": "input_audio", "_pcm": pcm, "_rate": 24000}]}
        run = NS(chunks=[{"text": "Already heard.", "end_ms": 200}], played_ms=200)
        _, observation, _ = await listener.classify(item, run)
        assert observation.is_final and observation.speech_ms == 9000
        assert observation.user_text == "The complete nine-second correction."
        assert [len(row[0]) for row in ear.calls] == [240 * 24, 400 * 24, 9 * 24000]
        np.testing.assert_array_equal(ear.calls[-1][0], pcm)
        part = item["content"][0]
        assert part["_pcm"] is pcm and "_rows" in part
        assert part["_transcript"] == "The complete nine-second correction."
    asyncio.run(fixture(check, ["No", "No", "The complete nine-second correction."]))


@pytest.mark.parametrize("mode,seconds,message", [("off", 9, "six-second"),
                                                   ("observe", 91, "90 seconds"),
                                                   ("semantic", 91, "90 seconds")])
def test_final_audio_caps_remain_bounded_without_truncation(mode, seconds, message):
    async def check(listener, service, session, ear, owner):
        session.settings["streaming_listener"] = mode
        item = {"id": "long", "content": [{"type": "input_audio", "_pcm": np.zeros(seconds * 16000),
                                            "_rate": 16000}]}
        with pytest.raises(ValueError, match=message):
            await listener.classify(item, NS(chunks=[], played_ms=0))
        assert not ear.calls and not service.jobs
        assert len(item["content"][0]["_pcm"]) == seconds * 16000
    asyncio.run(fixture(check))


def test_oversized_prefix_still_rejects_before_any_ear_or_brain_work():
    from dataclasses import replace

    async def check(listener, service, session, ear, owner):
        session.settings["streaming_listener"] = "semantic"
        oversized = replace(snapshot(), pcm=b"\0\0" * (9 * 24000))
        with pytest.raises(RuntimeError, match="oversized"):
            await listener.classify_prefix(oversized, heard_text="", pending_work=False, assistant_name="Frankie")
        assert not ear.calls and not service.jobs and not service.prompts
    asyncio.run(fixture(check))
