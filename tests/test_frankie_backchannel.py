"""Policy fixtures for an explicit English nod heuristic; no inference."""
from dataclasses import FrozenInstanceError

import pytest

from mtplx.frankie.backchannel import BackchannelLimits, decide_backchannel


def decide(text, *, voiced_ms=320, elapsed_ms=400, final=False, **kwargs):
    return decide_backchannel(text, voiced_ms=voiced_ms, elapsed_ms=elapsed_ms, final=final, **kwargs)


@pytest.mark.parametrize("text", [
    "mhm", "mm-hmm", "MM HMM.", "hmm", "hm", "hmmmm", "mmm", "mm", "uh huh",
    "uh-huh", "uhhuh", "um", "uh", "yeah", "yes", "yep", "yup", "okay", "ok",
    "right", "exactly", "Yeah, okay!", "mm-hmm, yeah", "Um, go on", "go on", "ＧＯ ＯＮ",
    "got it", "I see", "makes sense", "that makes sense", "yes that makes sense",
    "uh huh, got it", "yes, I see, that makes sense",
])
def test_only_brief_verbal_nods_preserve_current_playback(text):
    result = decide(text)
    assert result.action == "preserve" and result.reason == "brief_acknowledgment"
    assert not hasattr(result, "confidence")


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("text", [
    "yeah but", "mhm actually", "okay change that", "Um, I want", "uh no", "no", "stop",
    "Frankie", "Fr", "What is that", "Please wait", "I meant Thursday, not Friday",
    "yes please", "go on a trip", "I'm quoting stop talking", "Sam, stop talking to Alex",
    "I see why", "No, you're making sense, keep going", "The answer is yes", "hmm never mind",
])
def test_every_other_recognized_word_yields_without_stability_or_initial_grace(text, final):
    result = decide(text, voiced_ms=32, elapsed_ms=32, final=final)
    assert result.action == "yield"


@pytest.mark.parametrize("text", ["yeah?", "hmm？", '"yeah"', "‘mhm’", "«okay»", "'go on'"])
def test_questions_and_explicit_quoted_nods_are_not_preserved(text):
    assert decide(text).action == "yield"


@pytest.mark.parametrize("text", ["", "  ", "...", "…", "--", ", !"])
def test_empty_onset_waits_without_pausing_output(text):
    result = decide(text)
    assert result.action == "wait" and result.reason == "unrecognized_onset"


@pytest.mark.parametrize("voiced,elapsed,action", [
    (639, 959, "wait"), (640, 700, "wait"), (959, 1279, "wait"),
    (960, 1100, "wait"), (320, 1280, "wait"), (1599, 2399, "wait"),
    (1600, 2000, "yield"), (320, 2400, "yield"),
])
def test_empty_streaming_grace_has_both_voiced_and_elapsed_deadlines(voiced, elapsed, action):
    assert decide("", voiced_ms=voiced, elapsed_ms=elapsed).action == action


@pytest.mark.parametrize("voiced,elapsed,action", [
    (120, 400, "preserve"), (640, 960, "preserve"), (1600, 2400, "preserve"),
    (1601, 2000, "yield"), (320, 2401, "yield"),
])
def test_final_empty_short_input_is_a_possible_nonlexical_nod_but_still_bounded(voiced, elapsed, action):
    assert decide("", voiced_ms=voiced, elapsed_ms=elapsed, final=True).action == action


@pytest.mark.parametrize("text", ["hmmmm", "yeah yeah yeah", "um, go on"])
@pytest.mark.parametrize("voiced,elapsed,action", [(1600, 2400, "preserve"),
                                                (1601, 2000, "yield"), (320, 2401, "yield")])
def test_repeated_acknowledgments_do_not_suppress_sustained_user_speech(text, voiced, elapsed, action):
    assert decide(text, voiced_ms=voiced, elapsed_ms=elapsed).action == action


def test_full_current_text_must_replace_the_earlier_nod_observation():
    assert decide("yeah").action == "preserve"
    assert decide("yeah but could you explain").action == "yield"
    assert decide("yeah", final=True).action == "preserve"  # Pure, no cross-turn memory.


@pytest.mark.parametrize("prefix,complete", [
    ("go", "go on"), ("um go", "um go on"), ("got", "got it"), ("I", "I see"),
    ("makes", "makes sense"), ("that", "that makes sense"),
    ("that makes", "that makes sense"), ("yes that makes", "yes that makes sense"),
])
def test_exact_phrase_prefix_waits_within_total_cap_then_completed_phrase_preserves(prefix, complete):
    assert decide(prefix, voiced_ms=1300, elapsed_ms=1500).action == "wait"
    assert decide(complete, voiced_ms=1500, elapsed_ms=1800).action == "preserve"
    assert decide(prefix, voiced_ms=1300, elapsed_ms=1500, final=True).action == "yield"


@pytest.mark.parametrize("text", ["um go away", "got another question", "I need help", "makes me wonder",
                                 "that sounds wrong", "that makes no sense", "yes that makes no sense",
                                 "go on but", "I see but", "yes that makes sense except"])
def test_phrase_prefix_grace_never_accepts_a_word_outside_the_exact_sequences(text):
    assert decide(text, voiced_ms=32, elapsed_ms=32).action == "yield"


@pytest.mark.parametrize("voiced,elapsed", [(1601, 1900), (1300, 2401)])
def test_incomplete_acknowledgment_wait_is_bounded_by_total_duration(voiced, elapsed):
    result = decide("um go", voiced_ms=voiced, elapsed_ms=elapsed)
    assert result.action == "yield" and result.reason == "acknowledgment_too_long"


def test_bare_uh_remains_a_complete_allowed_filler_and_uh_huh_can_finish_it():
    assert decide("uh").action == "preserve"
    assert decide("uh", final=True).action == "preserve"
    assert decide("uh huh").action == "preserve"
    assert decide("uh but").action == "yield"


@pytest.mark.parametrize("text", ["umhm", "umhem", "umhum", "uhhum", "Umhem, go on.", "Aha.", "Ahuh."])
@pytest.mark.parametrize("final", [False, True])
def test_narrow_phonetic_nod_spellings_preserve_including_complete_go_on(text, final):
    assert decide(text, voiced_ms=1300, elapsed_ms=1500, final=final).action == "preserve"


@pytest.mark.parametrize("text", ["Im", "I hope", "I'm", "umhem but", "uhhum change that",
                                 "umhemgoon", "umhello", "Amen", "Aen", "O then", "aha but", "ahuh stop"])
@pytest.mark.parametrize("final", [False, True])
def test_phonetic_nod_variants_do_not_whitelist_mistaken_words_or_other_content(text, final):
    assert decide(text, final=final).action == "yield"


def test_unrecognized_nod_gets_a_bounded_recovery_window_after_the_old_fallback_point():
    assert decide("", voiced_ms=640, elapsed_ms=896).action == "wait"
    assert decide("Umhem, go on.", voiced_ms=1300, elapsed_ms=1500, final=True).action == "preserve"
    # This extra time is for absent/unstable ASR only, never recognized requests.
    assert decide("change that", voiced_ms=32, elapsed_ms=32).action == "yield"
    assert decide("", voiced_ms=960, elapsed_ms=1216).action == "wait"
    assert decide("", voiced_ms=1600, elapsed_ms=2000).action == "yield"
    assert decide("", voiced_ms=800, elapsed_ms=2400).action == "yield"


def test_oversized_text_is_bounded_before_normalization_and_tokenization():
    result = decide("yeah " * 10000)
    assert result.action == "yield" and result.reason == "text_too_long"


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "320"])
def test_invalid_durations_are_not_silent_policy_decisions(bad):
    with pytest.raises(ValueError):
        decide("yeah", voiced_ms=bad)
    with pytest.raises(ValueError):
        decide("yeah", elapsed_ms=bad)


def test_limits_are_explicit_and_decisions_immutable():
    limits = BackchannelLimits(empty_voiced_ms=200, empty_elapsed_ms=300,
                               acknowledgment_voiced_ms=800, acknowledgment_elapsed_ms=1000)
    assert decide("", voiced_ms=200, elapsed_ms=250, limits=limits).action == "yield"
    assert decide("yeah", voiced_ms=801, elapsed_ms=900, limits=limits).action == "yield"
    with pytest.raises(FrozenInstanceError):
        decide("yeah").action = "yield"
    with pytest.raises(ValueError):
        BackchannelLimits(empty_voiced_ms=2000, acknowledgment_voiced_ms=1600)
    with pytest.raises(ValueError):
        BackchannelLimits(empty_voiced_ms=True)
    with pytest.raises(TypeError):
        decide(None)
    with pytest.raises(ValueError):
        decide("yeah", final=1)
