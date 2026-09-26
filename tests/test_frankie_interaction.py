"""Prefix-only listening decisions must never silently change audio playback."""

import json
from dataclasses import replace

import pytest

from mtplx.frankie.interaction import (
    DecisionGate,
    ListenerObservation,
    parse_compact_decision,
    parse_decision,
    semantic_messages,
)


def observation(**kwargs):
    values = {"utterance_id": "user-1", "revision": 1, "observed_ms": 500,
              "user_text": "No, make that Thursday.", "assistant_speaking": True,
              "is_final": True}
    return ListenerObservation(**(values | kwargs))


def decision(obs, action="adapt", confidence=0.95):
    return parse_decision(json.dumps({"action": action, "confidence": confidence,
                                     "reason": "The user revised the day."}), obs)


def test_default_is_observation_only_even_for_clear_correction():
    obs = observation()
    gate = DecisionGate()
    gate.observe(obs)
    event = gate.decide(decision(obs))
    assert event["eligible"] and event["effective_action"] == "adapt"
    assert event["apply"] is False
    assert event["observed_ms"] == 500


def test_async_result_from_older_prefix_cannot_interrupt():
    first = observation(is_final=False, user_text="No")
    second = replace(first, revision=2, observed_ms=700, user_text="No worries")
    gate = DecisionGate(observation_only=False)
    gate.observe(first)
    gate.observe(second)
    event = gate.decide(decision(first, "yield", 1.0))
    assert event["gate"] == "stale_observation"
    assert event["effective_action"] == "wait" and not event["apply"]
    assert first.utterance_id not in gate.committed


@pytest.mark.parametrize("action", ["continue", "adapt", "yield", "stop"])
def test_confidence_cannot_certify_an_unstable_prefix(action):
    obs = observation(is_final=False, user_text="No")
    gate = DecisionGate(observation_only=False)
    gate.observe(obs)
    event = gate.decide(decision(obs, action, 1.0))
    assert event["gate"] == "unstable_prefix"
    assert not event["apply"]


def test_agreeing_transcript_snapshots_allow_decision_before_end_of_turn():
    obs = observation(is_final=False, transcript_stable=True, user_speaking=True)
    gate = DecisionGate(observation_only=False)
    gate.observe(obs)
    assert gate.decide(decision(obs))["apply"]


def test_wait_never_means_pause_or_cancel():
    obs = observation()
    gate = DecisionGate(observation_only=False)
    gate.observe(obs)
    event = gate.decide(decision(obs, "wait"))
    assert not event["apply"] and not event["eligible"]


@pytest.mark.parametrize("action,expected", [
    ("continue", "assistant_not_speaking"), ("adapt", "nothing_to_adapt"),
])
def test_ordinary_user_turn_is_not_misclassified_as_overlap(action, expected):
    obs = observation(assistant_speaking=False)
    gate = DecisionGate(observation_only=False)
    gate.observe(obs)
    assert gate.decide(decision(obs, action))["gate"] == expected


def test_pending_work_can_be_revised_while_assistant_silent():
    obs = observation(assistant_speaking=False, pending_work=True)
    gate = DecisionGate(observation_only=False)
    gate.observe(obs)
    assert gate.decide(decision(obs))["apply"]


def test_empty_transcript_is_not_semantic_evidence():
    obs = observation(user_text=" ")
    gate = DecisionGate(observation_only=False)
    gate.observe(obs)
    assert gate.decide(decision(obs, "yield", 1))["gate"] == "no_semantic_evidence"


def test_low_confidence_and_duplicate_decisions_have_no_side_effect():
    obs = observation()
    gate = DecisionGate(observation_only=False)
    gate.observe(obs)
    assert gate.decide(decision(obs, confidence=0.5))["gate"] == "uncertain"
    assert gate.decide(decision(obs))["gate"] == "duplicate_decision"


def test_gate_memory_is_bounded_and_evicted_results_are_stale():
    gate = DecisionGate(max_utterances=2)
    first = observation()
    gate.observe(first)
    gate.decide(decision(first))
    for i in range(2, 6):
        obs = observation(utterance_id=f"user-{i}")
        gate.observe(obs)
        gate.decide(decision(obs))
    assert len(gate.latest) == len(gate.committed) == 2
    assert gate.decide(decision(first))["gate"] == "stale_observation"
    assert len(gate.committed) == 2


def test_time_and_revision_must_advance():
    gate = DecisionGate()
    first = observation()
    gate.observe(first)
    for obs in (first, replace(first, revision=2, observed_ms=400)):
        with pytest.raises(ValueError, match="advance"):
            gate.observe(obs)


def test_prompt_is_bounded_and_contains_only_supplied_prefix():
    obs = observation(user_text="No, Thursday", assistant_heard_text="Friday works.")
    messages = semantic_messages(obs)
    assert json.loads(messages[1]["content"])["user_prefix"] == "No, Thursday"
    assert "tomorrow" not in messages[1]["content"]
    long = semantic_messages(replace(obs, user_text="a" * 100_000,
                                    assistant_heard_text="b" * 100_000))
    payload = json.loads(long[1]["content"])
    assert len(payload["user_prefix"]) <= 1200
    assert len(payload["assistant_already_heard"]) <= 1600
    assert "omitted" in payload["user_prefix"]


@pytest.mark.parametrize("raw", [
    "null", "[]", "{}", "<think>stop</think>",
    '{"action":"pause","confidence":1,"reason":"x"}',
    '{"action":"yield","confidence":NaN,"reason":"x"}',
    '{"action":"yield","confidence":true,"reason":"x"}',
    '{"action":"yield","confidence":-1,"reason":"x"}',
    '{"action":"yield","confidence":1,"reason":""}',
    '{"action":"yield","confidence":1,"reason":"x","apply":true}',
])
def test_malformed_model_output_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_decision(raw, observation())


@pytest.mark.parametrize("values", [
    {"revision": True}, {"observed_ms": -1}, {"speech_ms": 0.1},
    {"source": "future_audio"}, {"is_final": "yes"}, {"user_text": None},
    {"assistant_name": ""}, {"assistant_name": 42}, {"assistant_name": "x" * 101},
])
def test_invalid_observation_is_rejected(values):
    with pytest.raises(ValueError):
        observation(**values)


def test_compact_decisions_do_not_invent_confidence_or_enable_control():
    obs = observation()
    gate = DecisionGate(observation_only=False)
    gate.observe(obs)
    result = parse_compact_decision("adapt\n", obs)
    assert result.confidence is None
    event = gate.decide(result)
    assert event["action"] == "adapt" and event["gate"] == "uncalibrated"
    assert not event["apply"]
    prompt = semantic_messages(obs, compact=True)[0]["content"]
    assert "exactly one word" in prompt and "JSON" not in prompt


@pytest.mark.parametrize("raw", ["adapt.", "Adapt", "adapt because", "yield\nwait", "", None])
def test_compact_parser_requires_exact_action(raw):
    with pytest.raises(ValueError):
        parse_compact_decision(raw, observation())


def test_explicit_unscored_experiment_still_requires_stable_evidence():
    gate = DecisionGate(observation_only=False, allow_unscored=True)
    first = observation(is_final=False)
    gate.observe(first)
    assert gate.decide(parse_compact_decision("adapt", first))["gate"] == "unstable_prefix"
    stable = replace(first, revision=2, observed_ms=600, transcript_stable=True)
    gate.observe(stable)
    event = gate.decide(parse_compact_decision("adapt", stable))
    assert event["apply"] and event["confidence"] is None


def test_allow_unscored_does_not_turn_off_observation_only_default():
    gate = DecisionGate(allow_unscored=True)
    obs = observation()
    gate.observe(obs)
    event = gate.decide(parse_compact_decision("adapt", obs))
    assert event["eligible"] and not event["apply"]


def test_assistant_identity_is_explicit_and_unknown_by_default():
    unknown = json.loads(semantic_messages(observation())[1]["content"])
    known = json.loads(semantic_messages(observation(assistant_name="Aster"))[1]["content"])
    assert unknown["assistant_name"] is None
    assert known["assistant_name"] == "Aster"
