"""Fast English acknowledgment heuristic, not a semantic intent classifier.

Call only for VAD-qualified overlapping speech. Durations describe actual voiced
input and elapsed utterance time, excluding seeded history/pre-roll. Complete
current ASR text is required: never crop it to an earlier acknowledgment. No
inference, history mutation, confidence estimate, or output action occurs here.

The deliberate bias is to yield for anything beyond brief verbal nods, including
questions, quotations and speech addressed to another person. ASR mistakes can
still turn a nod into a word, or miss an interruption; this is not speaker or
addressee recognition. The caller must separately enforce response ownership.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class BackchannelLimits:
    # Unclear audio can still be a nod until the same brief-nod ceiling.
    # Recognized non-nods do not wait for either duration limit.
    empty_voiced_ms: int = 1600
    empty_elapsed_ms: int = 2400
    acknowledgment_voiced_ms: int = 1600
    acknowledgment_elapsed_ms: int = 2400
    max_text_chars: int = 256

    def __post_init__(self):
        if (any(type(value) is not int or value <= 0 for value in (
                self.empty_voiced_ms, self.empty_elapsed_ms, self.acknowledgment_voiced_ms,
                self.acknowledgment_elapsed_ms, self.max_text_chars))
                or self.empty_voiced_ms > self.acknowledgment_voiced_ms
                or self.empty_elapsed_ms > self.acknowledgment_elapsed_ms):
            raise ValueError("Backchannel limits must be positive, with bounded onset grace.")


@dataclass(frozen=True)
class BackchannelDecision:
    action: str  # preserve / wait / yield; wait also leaves current output alone
    reason: str


DEFAULT_LIMITS = BackchannelLimits()
_NODS = frozenset({"yes", "yeah", "yep", "yup", "okay", "ok", "right", "exactly", "um", "uh",
                  "umhm", "umhem", "umhum", "uhhum", "aha", "ahuh"})
_PHRASES = (("go", "on"), ("uh", "huh"), ("got", "it"), ("i", "see"),
            ("makes", "sense"), ("that", "makes", "sense"))
_TOKENS = re.compile(r"[a-z]+|[^a-z\s.,!…-]")


def _nod_status(tokens):
    index = 0
    while index < len(tokens):
        word = tokens[index]
        remaining = tuple(tokens[index:])
        matched = next((phrase for phrase in _PHRASES if remaining[:len(phrase)] == phrase), None)
        if matched is not None:
            index += len(matched)
            continue
        hum = len(word) >= 2 and "m" in word and all(letter in "mh" for letter in word)
        if word not in _NODS and not hum and word != "uhhuh":
            if any(len(remaining) < len(phrase) and phrase[:len(remaining)] == remaining
                   for phrase in _PHRASES):
                return "prefix"
            return "other"
        index += 1
    return "complete"


def decide_backchannel(transcript: str, *, voiced_ms: float, elapsed_ms: float,
                       final: bool = False, limits: BackchannelLimits = DEFAULT_LIMITS) -> BackchannelDecision:
    """Preserve brief nods, wait briefly for ASR, otherwise yield immediately.

    Unknown words never get an onset grace or require lexical stability. A
    trailing prefix of a small exact acknowledgment phrase may wait within the
    total acknowledgment duration caps. This also delays real requests starting
    with such a prefix until their next word or the cap; no continuation outside
    those phrases is accepted. Completed short empty input is allowed as an
    untranscribed nod/noise. Continued empty speech cannot wait indefinitely.
    """
    if not isinstance(transcript, str):
        raise TypeError("Use the complete current ASR transcript as text.")
    if (type(final) is not bool or any(isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in (voiced_ms, elapsed_ms))):
        raise ValueError("Use finite nonnegative durations and a boolean final flag.")
    if len(transcript) > limits.max_text_chars:
        return BackchannelDecision("yield", "text_too_long")
    text = unicodedata.normalize("NFKC", transcript).casefold().strip()
    if "?" in text:
        return BackchannelDecision("yield", "question")
    if any(mark in text for mark in ('"', "'", "“", "”", "‘", "’", "«", "»")):
        return BackchannelDecision("yield", "quoted_or_other_speech")
    tokens = _TOKENS.findall(text)
    short = (voiced_ms <= limits.acknowledgment_voiced_ms
             and elapsed_ms <= limits.acknowledgment_elapsed_ms)
    if not tokens:
        if final and short:
            return BackchannelDecision("preserve", "completed_nonlexical")
        if voiced_ms < limits.empty_voiced_ms and elapsed_ms < limits.empty_elapsed_ms:
            return BackchannelDecision("wait", "unrecognized_onset")
        return BackchannelDecision("yield", "unrecognized_speech")
    status = _nod_status(tokens)
    if status == "other":
        return BackchannelDecision("yield", "recognized_speech")
    if not short:
        return BackchannelDecision("yield", "acknowledgment_too_long")
    if status == "prefix":
        return BackchannelDecision("yield" if final else "wait", "incomplete_acknowledgment")
    return BackchannelDecision("preserve", "brief_acknowledgment")
