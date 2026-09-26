"""Bounded causal cutoffs: lifecycle tests, not intent/ASR qualification."""
import asyncio

import pytest
from test_frankie_background_session import setup, wait_for
from test_frankie_listening import listener, pcm, requested, stable, take_floor
from test_frankie_regeneration import events, staged_reply
from test_frankie_streaming_session import Listener, clear_events, enable, feed

from mtplx.frankie.listening import ListeningDecision, PrefixEvidence


def revalidation(core, ticket, text="Actually, change that to Thursday."):
    core.append(pcm(320))
    fresh = core.refresh(ticket)
    # Newly growing final words do not need a second full CTC plateau. The
    # original positive did; the second semantic decision sees all fresh words.
    return PrefixEvidence(fresh, text, "Actually, change that", fresh.samples - 3840)


def test_second_positive_can_yield_during_continuous_input_with_explicit_cutoff():
    core, clock = listener(allow_control=True)
    ticket = requested(core)
    original = stable(ticket, "Actually, change that.")
    fresh = revalidation(core, ticket)
    core.append(pcm(320, value=456))
    clock.now += .5
    assert not fresh.stable(160)
    result = core.complete(ticket, original, take_floor("yield"), recheck=fresh,
                           revalidated=take_floor("yield"))
    assert result.apply and result.reason == "causal_cutoff"
    assert result.observed_samples == fresh.snapshot.samples
    assert result.observed_samples < core.retained_bytes // 2
    assert core.inflight is None and core.request() is None


@pytest.mark.parametrize("text,action", [
    ("Actually, don't change that. Keep going.", "continue"),
    ("I'm quoting a line: actually change that.", "wait"),
    ("Ravi, actually change that. Frankie, keep explaining.", "continue"),
    ("Please stop talking is the phrase I'm reading aloud.", "wait"),
])
def test_known_revision_requires_second_full_semantics_and_can_block_old_yield(text, action):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    fresh = revalidation(core, ticket, text)
    core.append(pcm(100))
    result = core.complete(ticket, stable(ticket), take_floor("yield"), recheck=fresh,
                           revalidated=ListeningDecision(action))
    assert not result.apply and result.action == action
    assert result.reason == ("backchannel" if action == "continue" else "wait")


def test_revalidation_never_bootstraps_an_unstable_initial_positive():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    initial = PrefixEvidence(ticket, "Actually, change that", "Actually", ticket.samples - 3840)
    fresh = revalidation(core, ticket)
    result = core.complete(ticket, initial, take_floor("yield"), recheck=fresh,
                           revalidated=take_floor("yield"))
    assert not result.apply and result.reason == "unstable_evidence"


@pytest.mark.parametrize("elapsed,apply", [(0, True), (1, True), (1.001, False),
                                         (-.001, False), (float("nan"), False)])
def test_revalidated_cutoff_has_one_second_wall_age_bound(elapsed, apply):
    core, clock = listener(allow_control=True)
    ticket = requested(core)
    fresh = revalidation(core, ticket)
    clock.now = fresh.snapshot.captured_at + elapsed
    result = core.complete(ticket, stable(ticket), take_floor("yield"), recheck=fresh,
                           revalidated=take_floor("yield"))
    assert result.apply is apply
    assert result.reason == ("causal_cutoff" if apply else "expired")


@pytest.mark.parametrize("audio_ms,apply", [(1000, True), (1001, False)])
def test_unreviewed_pcm_also_has_one_second_bound_even_when_uploaded_instantly(audio_ms, apply):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    fresh = revalidation(core, ticket)
    core.append(pcm(audio_ms))
    result = core.complete(ticket, stable(ticket), take_floor("yield"), recheck=fresh,
                           revalidated=take_floor("yield"))
    assert result.apply is apply
    assert result.reason == ("causal_cutoff" if apply else "causal_audio_lag")


@pytest.mark.parametrize("change,reason", [("cancel", "stale_ticket"), ("finish", "stale_ticket"),
                                         ("refresh", "wrong_recheck"), ("overflow", "audio_budget")])
def test_revalidation_keeps_epoch_snapshot_and_audio_budget_guards(change, reason):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    fresh = revalidation(core, ticket)
    if change == "cancel":
        core.cancel()
    elif change == "finish":
        core.finish_input()
    elif change == "refresh":
        core.refresh(ticket)
    else:
        core.append(pcm(6000))
    result = core.complete(ticket, stable(ticket), take_floor("yield"), recheck=fresh,
                           revalidated=take_floor("yield"))
    assert not result.apply and result.reason == reason


@pytest.mark.parametrize("initial", [ListeningDecision("wait"), ListeningDecision("yield"),
                                     ListeningDecision("yield", True, False)])
def test_revalidation_cannot_bypass_initial_semantic_qualification(initial):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    fresh = revalidation(core, ticket)
    result = core.complete(ticket, stable(ticket), initial, recheck=fresh,
                           revalidated=take_floor("yield"))
    assert not result.apply and result.reason == "invalid_revalidation"


def test_observe_mode_causal_revalidation_is_still_observation_only():
    core, _ = listener()
    ticket = requested(core)
    fresh = revalidation(core, ticket)
    core.append(pcm(100))
    result = core.complete(ticket, stable(ticket), take_floor("yield"), recheck=fresh,
                           revalidated=take_floor("yield"))
    assert result.eligible and not result.apply and result.reason == "observation_only"


class SecondObservation(Listener):
    def __init__(self, *, action="yield", text="Actually, change the date to Thursday."):
        super().__init__(held=True, recheck_text=text)
        self.second_action, self.second_text = action, text
        self.second_entered, self.second_release = asyncio.Event(), asyncio.Event()

    async def classify_prefix(self, snapshot, **kwargs):
        if not self.snapshots:
            return await super().classify_prefix(snapshot, **kwargs)
        self.snapshots.append(snapshot)
        self.contexts.append(kwargs)
        self.second_entered.set()
        await asyncio.wait_for(self.second_release.wait(), 2)
        decision = (take_floor("yield") if self.second_action == "yield"
                    else ListeningDecision(self.second_action))
        evidence = PrefixEvidence(snapshot, self.second_text, "Actually, change the date", snapshot.samples - 3840)
        return decision, evidence, {"second_fixture": True}


async def start_second(session, backend):
    await feed(session, 16000, 10)
    await asyncio.wait_for(backend.entered.wait(), 2)
    await feed(session, 17000, 2)
    backend.release.set()
    await asyncio.wait_for(backend.second_entered.wait(), 2)


def test_live_session_revalidates_once_then_yields_while_preserving_growing_input():
    async def check(session, engine):
        backend = SecondObservation()
        await enable(session, backend)
        old = staged_reply(session)
        await start_second(session, backend)
        await feed(session, 18000, 4)
        backend.second_release.set()
        await wait_for(lambda: session.prefix_task is None)
        assert old.abort.is_set() and session.listening and not engine.calls
        assert len(backend.snapshots) == 2 and len(backend.rechecks) == 1
        assert len(session.frames) == 16
        observed = [event for event in events(session) if event["type"] == "frankie.interaction.prefix"][-1]
        assert observed["apply"] and observed["gate"] == "causal_cutoff" and observed["causal_cutoff"]
        assert observed["decision_audio_ms"] == round(backend.snapshots[1].observed_ms)
        assert 0 <= observed["apply_lag_ms"] <= 1000
        assert observed["recheck_transcript"] == backend.second_text
        assert observed["metrics"]["revalidation"] == {"second_fixture": True}
        await feed(session, 0, 10)
        await wait_for(lambda: len(engine.calls) == 1)
        audio = next(item for item in engine.calls[0] if item.get("role") == "user")
        assert audio["content"][0]["_pcm"].size == 26 * 768
    asyncio.run(setup(check))


@pytest.mark.parametrize("action,text", [
    ("continue", "Actually, don't change the date. Continue."),
    ("wait", "I'm quoting: actually change the date."),
    # Same lexical words as the initial positive; newly recognized quotation
    # punctuation must still be presented to the full semantic classifier.
    ("wait", '"Actually, change the date."'),
])
def test_live_session_second_semantics_can_veto_known_negation_or_quote(action, text):
    async def check(session, engine):
        backend = SecondObservation(action=action, text=text)
        await enable(session, backend)
        old = staged_reply(session)
        await start_second(session, backend)
        backend.second_release.set()
        await wait_for(lambda: session.prefix_task is None)
        assert not old.abort.is_set() and not engine.calls and clear_events(session) == []
        assert len(backend.snapshots) == 2 and len(backend.rechecks) == 1
    asyncio.run(setup(check))


def test_superseded_response_cannot_be_cleared_by_a_second_classifier_result():
    async def check(session, engine):
        backend = SecondObservation()
        await enable(session, backend)
        old = staged_reply(session)
        await start_second(session, backend)
        replacement = staged_reply(session)
        backend.second_release.set()
        await wait_for(lambda: session.prefix_task is None)
        assert session.current is replacement and not replacement.abort.is_set()
        assert not old.abort.is_set() and not engine.calls and clear_events(session) == []
        assert len(backend.snapshots) == 2 and len(backend.rechecks) == 1
    asyncio.run(setup(check))


def test_cancel_during_second_classifier_cannot_clear_or_reserve_old_response():
    async def check(session, engine):
        backend = SecondObservation()
        await enable(session, backend)
        old = staged_reply(session)
        await start_second(session, backend)
        await session.handle({"type": "input_audio_buffer.clear"})
        await wait_for(lambda: session.prefix_task is None)
        assert session.prefix_listener.inflight is None
        assert not old.abort.is_set() and not engine.calls and clear_events(session) == []
    asyncio.run(setup(check))
