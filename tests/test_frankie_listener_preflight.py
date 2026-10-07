"""Fake-owner acoustic screening and pure semantic budget accounting."""
import asyncio
import threading

import pytest
from test_frankie_background_session import setup, wait_for
from test_frankie_listening import listener, pcm, requested, stable
from test_frankie_prefix_adapter import fixture, floor_payload, snapshot
from test_frankie_regeneration import staged_reply
from test_frankie_streaming_session import Listener, clear_events, enable, feed

from mtplx.frankie.listening import ListeningDecision, PrefixEvidence


@pytest.mark.parametrize("older,current,reason", [("", "", "empty"), ("...", "...", "empty"),
                                                  ("", "Actually", "unstable"),
                                                  ("Stop", "Stop worrying.", "unstable")])
@pytest.mark.parametrize("mtp", [0, 2])
def test_empty_or_unstable_acoustics_never_warm_or_run_brain(older, current, reason, mtp):
    async def check(backend, service, session, ear, owner):
        def forbidden(*args, **kwargs):
            pytest.fail("An ineligible acoustic observation must not prepare the brain.")
        service.warm_listener = service.insert = forbidden
        decision, evidence, stats = await backend.classify_prefix(
            snapshot(), heard_text="", pending_work=False, assistant_name="Frankie")
        assert decision == ListeningDecision("wait") and evidence.text == current
        assert stats["semantic_skipped"] and stats["semantic_skip_reason"] == reason
        assert "completion_tokens" not in stats and service.listener_bank is None
        assert stats["listener_owner_slices"] == 1
        assert stats["listener_owner_work_ms"] >= 0
        assert len(ear.calls) == 2 and not service.jobs and backend.job is None
    asyncio.run(fixture(check, [older, current], mtp=mtp, real_service=True))


def test_stable_preflight_and_brain_use_same_owner_and_complete_text_once():
    async def check(backend, service, session, ear, owner):
        original_prepare, brain_threads = service.prepare, []
        def prepare(job):
            brain_threads.append(threading.get_ident())
            return original_prepare(job)
        service.prepare = prepare
        submit, admitted = service.submit, []
        def tracked_submit(*args, **kwargs):
            job = submit(*args, **kwargs)
            admitted.append(job)
            return job
        service.submit = tracked_submit
        _, evidence, stats = await backend.classify_prefix(
            snapshot(), heard_text="The train route.", pending_work=True, assistant_name="Frankie")
        assert evidence.stable(160) and not stats["semantic_skipped"]
        assert len(service.prompts) == 1 and len(ear.calls) == 2
        assert brain_threads == [ear.calls[0][2]] == [ear.calls[1][2]]
        assert floor_payload(service.prompts[0])["user_prefix"] == "Stop worrying."
        assert stats["ear_ms"] >= 0 and stats["completion_tokens"] == 2
        assert stats["listener_owner_slices"] == 1
        assert len(admitted) == 1 and not admitted[0].prepare_only
    asyncio.run(fixture(check, ["Stop worrying.", "Stop worrying."]))


def test_revalidation_can_classify_complete_growing_text_without_second_plateau():
    async def check(backend, service, session, ear, owner):
        service.action = "yield"
        decision, evidence, stats = await backend.classify_prefix(
            snapshot(), heard_text="", pending_work=False, assistant_name="Frankie", require_stable=False)
        assert decision.action == "yield" and not stats["semantic_skipped"]
        assert evidence.previous_text is None and not evidence.stable(160)
        assert len(ear.calls) == 1 and len(service.prompts) == 1
        assert floor_payload(service.prompts[0])["user_prefix"] == "Actually, make the date Thursday."
    asyncio.run(fixture(check, ["Actually, make the date Thursday."]))


@pytest.mark.parametrize("supersede", ["cancel", "close"])
def test_supersession_during_ear_prevents_same_job_brain_preparation(supersede):
    async def check(backend, service, session, ear, owner):
        def after_ear():
            if len(ear.calls) == 2:
                if supersede == "cancel":
                    backend.cancel()
                else:
                    session.closed = True
        ear.hook = after_ear
        with pytest.raises(RuntimeError, match="cancelled|consumer"):
            await backend.classify_prefix(snapshot(), heard_text="", pending_work=False, assistant_name="Frankie")
        assert len(ear.calls) == 2 and not service.prompts and not service.jobs and backend.job is None
    asyncio.run(fixture(check, ["Please wait.", "Please wait."]))


def skipped(core, ticket):
    return core.complete(ticket, PrefixEvidence(ticket, ""), ListeningDecision("wait"), semantic_skipped=True)


def test_refunded_acoustic_checks_keep_spacing_and_leave_three_actual_semantic_calls():
    core, _ = listener()
    for _ in range(8):
        core.append(pcm(320))
        ticket = core.request()
        assert ticket is not None and skipped(core, ticket).reason == "semantic_skipped"
        assert core.request() is None  # Refund must not bypass the acoustic interval.
    for _ in range(3):
        core.append(pcm(320))
        ticket = core.request()
        assert ticket is not None
        core.complete(ticket, stable(ticket), ListeningDecision("wait"))
    core.append(pcm(320))
    assert core.request() is None


def test_refunds_cannot_create_unbounded_acoustic_work():
    core, _ = listener()
    checks = 0
    while core.append(pcm(320)):
        ticket = core.request()
        assert ticket is not None
        checks += 1
        skipped(core, ticket)
    assert checks == 18 and core.exhausted and core.request() is None
    assert core.retained_bytes == 6000 * 24 * 2


def test_optional_revalidation_counts_against_same_three_call_budget():
    core, _ = listener()
    first = requested(core)
    assert core.reserve_revalidation(first)
    assert not core.reserve_revalidation(first)  # No recursive third classification.
    core.complete(first, stable(first), ListeningDecision("wait"))
    core.append(pcm(320))
    third = core.request()
    assert third is not None and not core.reserve_revalidation(third)
    core.complete(third, stable(third), ListeningDecision("wait"))
    core.append(pcm(320))
    assert core.request() is None


@pytest.mark.parametrize("invalid", ["stable", "positive", "assertions", "wrong_snapshot", "reserved", "not_bool"])
def test_refund_requires_valid_skipped_wait_and_actual_ineligible_acoustic_evidence(invalid):
    core, _ = listener(max_partial_probes=1 if invalid != "reserved" else 2)
    ticket = requested(core)
    evidence, decision, flag = PrefixEvidence(ticket, ""), ListeningDecision("wait"), True
    if invalid == "stable":
        evidence = stable(ticket)
    elif invalid == "positive":
        decision = ListeningDecision("yield", True, True)
    elif invalid == "assertions":
        decision = ListeningDecision("wait", True, True)
    elif invalid == "wrong_snapshot":
        evidence = PrefixEvidence(core.refresh(ticket), "")
    elif invalid == "reserved":
        assert core.reserve_revalidation(ticket)
    else:
        flag = 1
    result = core.complete(ticket, evidence, decision, semantic_skipped=flag)
    assert not result.apply and result.reason in {"invalid_semantic_skip", "wrong_evidence"}
    core.append(pcm(320))
    assert core.request() is None


def test_late_duplicate_skip_cannot_refund_a_newer_ticket():
    core, _ = listener(max_partial_probes=1)
    first = requested(core)
    skipped(core, first)
    core.append(pcm(320))
    current = core.request()
    assert skipped(core, first).reason == "stale_ticket" and core.inflight is current
    core.complete(current, stable(current), ListeningDecision("wait"))
    core.append(pcm(320))
    assert core.request() is None


def test_session_skipped_preflights_do_not_consume_the_later_semantic_opportunity():
    class EmptyFirst(Listener):
        async def classify_prefix(self, snapshot, **kwargs):
            if len(self.snapshots) < 4:
                self.snapshots.append(snapshot)
                return ListeningDecision("wait"), PrefixEvidence(snapshot, ""), {"semantic_skipped": True}
            return await super().classify_prefix(snapshot, **kwargs)

    async def check(session, engine):
        backend = EmptyFirst()
        await enable(session, backend)
        old = staged_reply(session)
        for count in range(1, 5):
            await feed(session, 16000, 10)
            await wait_for(lambda: session.prefix_task is None)
            assert len(backend.snapshots) == count
            assert not old.abort.is_set() and clear_events(session) == []
        await feed(session, 17000, 10)
        await wait_for(lambda: old.abort.is_set())
        assert len(backend.snapshots) == 5 and len(session.frames) == 50
        assert session.listening and not engine.calls
    asyncio.run(setup(check))
