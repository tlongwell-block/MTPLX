"""Pure listener lifecycle tests; no MLX, models, threads, or audio device."""
import struct
from dataclasses import FrozenInstanceError

import pytest

from mtplx.frankie.listening import (
    ListeningDecision,
    PrefixEvidence,
    StreamingListener,
)


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


def pcm(ms, rate=24000, value=123):
    return struct.pack("<h", value) * (ms * rate // 1000)


def listener(**kwargs):
    clock = Clock()
    core = StreamingListener(clock=clock, **kwargs)
    core.begin("response-a", "utterance-a")
    return core, clock


def stable(snapshot, text="Please stop talking."):
    return PrefixEvidence(snapshot, text, text, snapshot.samples - snapshot.sample_rate * 160 // 1000)


def take_floor(action="stop"):
    return ListeningDecision(action, addressed_to_assistant=True, sufficient_evidence=True)


def requested(core, ms=400):
    core.append(pcm(ms))
    ticket = core.request()
    assert ticket is not None
    return ticket


def test_default_is_observation_only_even_with_explicit_semantic_assertions():
    core, _ = listener()
    ticket = requested(core)
    result = core.complete(ticket, stable(ticket), take_floor())
    assert result.eligible and not result.apply
    assert result.reason == "observation_only" and result.intent == "take_floor"
    assert core.inflight is None


@pytest.mark.parametrize("action", ["continue", "wait"])
def test_backchannel_and_wait_never_pause_clear_or_consume_the_utterance(action):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    result = core.complete(ticket, stable(ticket, "Mm-hmm."), ListeningDecision(action))
    assert not result.eligible and not result.apply
    assert result.reason == ("backchannel" if action == "continue" else "wait")
    core.append(pcm(320))
    assert core.request() is not None  # Later correction can still claim the floor.


@pytest.mark.parametrize("action", ["stop", "adapt", "yield"])
def test_qualified_control_has_exact_identity_and_can_apply_only_once(action):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    result = core.complete(ticket, stable(ticket), take_floor(action))
    assert result.apply and result.eligible
    assert (result.response_id, result.utterance_id, result.epoch, result.revision) == (
        ticket.response_id, ticket.utterance_id, ticket.epoch, ticket.revision)
    assert result.observed_samples == ticket.samples
    assert not core.complete(ticket, stable(ticket), take_floor(action)).apply
    core.append(pcm(320))
    assert core.request() is None


@pytest.mark.parametrize("addressed", [None, False])
def test_compact_action_cannot_invent_addressee_evidence(addressed):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    result = core.complete(ticket, stable(ticket), ListeningDecision("stop", addressed, True))
    assert not result.apply and result.reason == "uncertain_addressee"


def test_compact_action_cannot_invent_sufficient_semantic_evidence():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    result = core.complete(ticket, stable(ticket), ListeningDecision("adapt", True))
    assert not result.apply and result.reason == "insufficient_evidence"


def test_stable_but_wrong_classifier_is_not_claimed_to_be_solved_by_this_gate():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    # Deliberately wrong backend assertion: this module cannot infer semantics.
    result = core.complete(ticket, stable(ticket, "No."), take_floor("adapt"))
    assert result.apply  # Demonstrates why policy qualification is separate.


def test_snapshot_is_disposable_immutable_owned_pcm_not_authoritative_features():
    core, _ = listener()
    raw = bytearray(pcm(400))
    core.append(raw)
    ticket = core.request()
    raw[:] = b"\0" * len(raw)
    assert ticket.pcm == pcm(400) and isinstance(ticket.pcm, bytes)
    with pytest.raises(FrozenInstanceError):
        ticket.pcm = b""
    core.append(pcm(200, value=456))
    assert ticket.samples == 9600 and ticket.pcm == pcm(400)
    assert not hasattr(ticket, "_rows") and not hasattr(ticket, "_transcript")
    refresh = core.refresh(ticket)
    assert refresh.pcm == pcm(400) + pcm(200, value=456)


@pytest.mark.parametrize("rate", [16000, 24000])
def test_actual_sample_end_alignment_includes_all_disposable_prefix_audio(rate):
    core, _ = listener()
    core.begin("r", "u", sample_rate=rate, start_ms=123)
    core.append(pcm(400, rate))
    snapshot = core.request()
    assert snapshot.samples == rate * 400 // 1000
    assert snapshot.observed_ms == 523


def test_one_inflight_and_latest_audio_coalescing_without_a_pending_job_queue():
    core, _ = listener()
    first = requested(core)
    for _ in range(8):
        core.append(pcm(40))
        assert core.request() is None
    core.retire(first)
    latest = core.request()
    assert latest.samples == 17280 and latest.revision == first.revision + 1
    assert core.inflight is latest


def test_new_unreviewed_audio_blocks_an_otherwise_good_old_decision():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    core.append(pcm(20))
    result = core.complete(ticket, stable(ticket), take_floor())
    assert not result.apply and result.reason == "newer_audio"


def test_same_owner_recheck_of_complete_unchanged_current_transcript_can_apply():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    core.append(pcm(320))
    latest = core.refresh(ticket)
    result = core.complete(ticket, stable(ticket), take_floor(),
                           recheck=PrefixEvidence(latest, "Please stop talking."))
    assert result.apply and result.observed_samples == latest.samples


@pytest.mark.parametrize("old,new", [
    ("Stop", "Stop worrying"), ("No", "No worries"),
    ("Stop talking", "Don't stop talking"),
    ("Please wait", "Maya, please wait"),
])
def test_newer_qualifier_is_not_cropped_away_to_reuse_a_stale_positive(old, new):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    core.append(pcm(320))
    latest = core.refresh(ticket)
    result = core.complete(ticket, stable(ticket, old), take_floor(),
                           recheck=PrefixEvidence(latest, new))
    assert not result.apply and result.reason == "changed_transcript"


def test_audio_arriving_after_recheck_still_requires_new_evidence():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    core.append(pcm(320))
    latest = core.refresh(ticket)
    core.append(pcm(20))
    result = core.complete(ticket, stable(ticket), take_floor(),
                           recheck=PrefixEvidence(latest, "Please stop talking."))
    assert not result.apply and result.reason == "newer_audio"


def test_superseded_recheck_object_cannot_be_mistaken_for_latest_evidence():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    old = core.refresh(ticket)
    core.refresh(ticket)
    result = core.complete(ticket, stable(ticket), take_floor(),
                           recheck=PrefixEvidence(old, "Please stop talking."))
    assert not result.apply and result.reason == "wrong_recheck"


@pytest.mark.parametrize("previous", ["Please stop", "Please", "", "Stop worrying"])
def test_partial_stability_requires_entire_transcript_agreement(previous):
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    evidence = PrefixEvidence(ticket, "Please stop talking", previous, ticket.samples - 3840)
    result = core.complete(ticket, evidence, take_floor())
    assert not result.apply and result.reason == "unstable_evidence"


def test_snapshots_too_close_together_do_not_prove_stability():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    evidence = PrefixEvidence(ticket, "Stop talking", "Stop talking", ticket.samples - 24)
    assert not core.complete(ticket, evidence, take_floor()).apply


def test_final_input_does_not_require_two_matching_transcript_snapshots():
    core, _ = listener(allow_control=True)
    core.append(pcm(120))
    core.finish_input()
    ticket = core.request()
    result = core.complete(ticket, PrefixEvidence(ticket, "Please wait."), take_floor())
    assert ticket.final and result.apply


def test_finalization_invalidates_partial_but_holds_owner_slot_until_retirement():
    core, _ = listener(allow_control=True)
    partial = requested(core)
    assert core.finish_input() is partial
    assert core.request() is None and core.refresh(partial) is None
    result = core.complete(partial, stable(partial), take_floor())
    assert not result.apply and result.reason == "stale_ticket"
    final = core.request()
    assert final.final and final.epoch != partial.epoch
    assert final.pcm == partial.pcm
    assert core.complete(final, PrefixEvidence(final, "Stop talking."), take_floor()).apply


def test_cancel_and_new_turn_cannot_enqueue_behind_an_unretired_old_owner_job():
    core, _ = listener(allow_control=True)
    old = requested(core)
    core.cancel()
    assert core.inflight is old and core.retained_bytes == 0
    core.begin("response-b", "utterance-b")
    core.append(pcm(500))
    assert core.request() is None and core.refresh(old) is None
    result = core.complete(old, stable(old), take_floor())
    assert not result.apply and result.reason == "stale_ticket"
    new = core.request()
    assert new.response_id == "response-b"
    core.complete(old, stable(old), take_floor())  # Duplicate old callback.
    assert core.inflight is new


def test_cancelled_or_failed_callback_finally_retirement_releases_cross_turn_slot():
    core, _ = listener()
    old = requested(core)
    core.cancel()
    core.begin("response-b", "utterance-b")
    core.append(pcm(400))
    try:
        raise RuntimeError("Cancelled owner callback")
    except RuntimeError:
        pass
    finally:
        core.retire(old)
    assert core.request().response_id == "response-b"


def test_reusing_ids_still_invalidates_previous_epoch():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    core.begin(ticket.response_id, ticket.utterance_id)
    result = core.complete(ticket, stable(ticket), take_floor())
    assert not result.apply and result.reason == "stale_ticket"


def test_partial_probe_budget_and_interval_still_reserve_a_final_opportunity():
    core, _ = listener(max_partial_probes=2)
    first = requested(core)
    core.retire(first)
    assert core.request() is None
    core.append(pcm(319))
    assert core.request() is None
    core.append(pcm(1))
    second = core.request()
    core.retire(second)
    core.append(pcm(500))
    assert core.request() is None
    core.finish_input()
    final = core.request()
    assert final.final and final.samples == 1220 * 24
    core.retire(final)
    assert core.request() is None


def test_audio_budget_disables_observer_without_acoustically_taking_the_floor():
    core, _ = listener(allow_control=True)
    ticket = requested(core)
    assert core.append(pcm(5600)) and core.retained_bytes == 6000 * 24 * 2
    assert not core.append(pcm(20))
    assert core.exhausted and core.retained_bytes == 6000 * 24 * 2
    result = core.complete(ticket, stable(ticket), take_floor())
    assert not result.apply and result.reason == "audio_budget"
    assert core.request() is None
    core.finish_input()
    assert core.request() is None  # Host keeps full input; never truncate it to fit.


@pytest.mark.parametrize("elapsed", [2.001, -0.01, float("nan"), float("inf")])
def test_delayed_or_invalid_time_cannot_revalidate_an_old_result(elapsed):
    core, clock = listener(allow_control=True)
    ticket = requested(core)
    clock.now += elapsed
    result = core.complete(ticket, stable(ticket), take_floor())
    assert not result.apply and result.reason == "expired"
    assert core.inflight is None


def test_empty_final_evidence_is_not_semantic_evidence():
    core, _ = listener(allow_control=True)
    core.append(pcm(120))
    core.finish_input()
    ticket = core.request()
    result = core.complete(ticket, PrefixEvidence(ticket, "..."), take_floor())
    assert not result.apply and result.reason == "empty_evidence"


@pytest.mark.parametrize("kwargs", [
    {"allow_control": 1}, {"max_partial_probes": 0}, {"min_audio_ms": 0},
    {"max_audio_ms": 6001}, {"max_result_age_ms": -1}, {"stability_ms": 400},
])
def test_invalid_bounds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        StreamingListener(**kwargs)


def test_invalid_audio_never_advances_the_snapshot():
    core, _ = listener()
    with pytest.raises(ValueError):
        core.append(b"\x00")
    assert core.retained_bytes == 0 and core.request() is None
    core.finish_input()
    with pytest.raises(ValueError):
        core.append(pcm(1))


@pytest.mark.parametrize("kwargs", [
    {"action": "pause"}, {"action": "stop", "addressed_to_assistant": 1},
    {"action": "stop", "sufficient_evidence": "yes"},
])
def test_malformed_semantic_assertions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ListeningDecision(**kwargs)


def test_incomplete_or_unaligned_comparison_evidence_is_rejected():
    core, _ = listener()
    ticket = requested(core)
    for previous, endpoint in [("Stop", None), (None, 100), ("Stop", ticket.samples),
                               ("Stop", 0), ("Stop", True)]:
        with pytest.raises(ValueError):
            PrefixEvidence(ticket, "Stop", previous, endpoint)
