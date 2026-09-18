"""Bounded v5 listener lifecycle; no inference, playback, or history mutations.

Call the mutable lifecycle on the session's event-loop thread. Pass its immutable
snapshots to the *existing* ear/brain inference owner, never to a second model or
inference executor. Partial ear features belong to the disposable observation,
not to authoritative conversation parts. This module deliberately retains no
features, transcripts, classifier cache, or vocabulary of interruption words.

Lexical stability is not intent calibration. The semantic callback must establish
addressee and sufficient floor-taking evidence; the gate cannot make an unreliable
classifier reliable. Control is disabled by default. Its tests prove lifecycle
invariants, not the accuracy of any learned or prompted semantic policy.
"""
from __future__ import annotations

import math
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from .interaction import ACTIONS, MAX_OBSERVATION_SECONDS


def _words(text: str) -> tuple[str, ...]:
    # Compare all recognized words, never a longest common prefix. Keep the
    # original complete text for semantics, including punctuation and qualifiers.
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return tuple(re.findall(r"\b[\w']+\b", text))


@dataclass(frozen=True)
class ListeningSnapshot:
    response_id: str
    utterance_id: str
    epoch: int
    revision: int
    pcm: bytes  # owned, immutable, mono little-endian PCM16 from utterance onset
    sample_rate: int
    start_ms: int
    final: bool
    captured_at: float  # monotonic host time, not an inferred word timestamp

    @property
    def samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def observed_ms(self) -> float:
        return self.start_ms + 1000 * self.samples / self.sample_rate


@dataclass(frozen=True)
class PrefixEvidence:
    """Actual full transcripts from two independently re-encoded PCM prefixes.

    ``previous_samples`` refers to an earlier endpoint in this same snapshot.
    It is not a word timestamp. Never replace ``text`` with the common prefix or
    a clean fixture script. The previous transcript cannot certify semantics.
    """
    snapshot: ListeningSnapshot
    text: str
    previous_text: str | None = None
    previous_samples: int | None = None

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("Listener evidence must contain actual transcript text.")
        if (self.previous_text is None) != (self.previous_samples is None):
            raise ValueError("A comparison transcript requires its PCM endpoint.")
        if self.previous_text is not None and (
            not isinstance(self.previous_text, str)
            or type(self.previous_samples) is not int
            or not 0 < self.previous_samples < self.snapshot.samples
        ):
            raise ValueError("Comparison evidence must end inside the current PCM prefix.")

    def stable(self, separation_ms: int) -> bool:
        return (
            self.previous_text is not None
            and bool(_words(self.text))
            and _words(self.text) == _words(self.previous_text)
            and (self.snapshot.samples - self.previous_samples) * 1000
            >= separation_ms * self.snapshot.sample_rate
        )

    def same_transcript(self, other: PrefixEvidence) -> bool:
        return _words(self.text) == _words(other.text)

    @property
    def has_words(self) -> bool:
        return bool(_words(self.text))


@dataclass(frozen=True)
class ListeningDecision:
    """Semantic callback result, not a probability or an acoustic heuristic.

    Reuse the existing continue/adapt/yield/stop/wait vocabulary. ``continue`` is
    a backchannel; adapt/yield/stop claim the floor. The callback must explicitly
    establish that this is assistant-directed and already sufficient to change
    the floor. A request's *effect* and subsequent answer remain Session's job.
    These booleans are backend assertions, not proof of a calibrated classifier.
    """
    action: str
    addressed_to_assistant: bool | None = None
    sufficient_evidence: bool = False

    def __post_init__(self):
        if not isinstance(self.action, str) or self.action not in ACTIONS:
            raise ValueError("Unknown semantic listening action.")
        if self.addressed_to_assistant is not None and type(self.addressed_to_assistant) is not bool:
            raise ValueError("Addressee must be true, false, or unknown.")
        if type(self.sufficient_evidence) is not bool:
            raise ValueError("Evidence sufficiency must be a boolean.")

    @property
    def intent(self) -> str:
        if self.action == "continue":
            return "backchannel"
        return "wait" if self.action == "wait" else "take_floor"


# Invoke only on the existing inference owner, with bounded snapshots/evidence.
# The core intentionally neither invokes this callback nor creates a worker.
SemanticCallback = Callable[[PrefixEvidence], ListeningDecision]


@dataclass(frozen=True)
class ListeningResult:
    response_id: str
    utterance_id: str
    epoch: int
    revision: int
    observed_samples: int
    action: str
    intent: str
    reason: str
    eligible: bool
    apply: bool


class StreamingListener:
    """One active utterance, one in-flight ticket, no pending inference queue.

    Append into the bounded onset-preserving buffer while a job runs. The next
    ``request`` coalesces all current PCM. Cancellation invalidates old evidence
    but does not free its owner reservation until ``retire``/``complete``; callers
    must cancel and retire the actual job in ``finally`` even after disconnect.

    If audio advances while the brain classifies, ``refresh(ticket)`` provides
    a new disposable snapshot for a same-owner ear recheck. ``complete`` accepts
    it only when the *full* recognized transcript is unchanged and no further
    unreviewed PCM exists. One explicit semantic revalidation can instead use a
    bounded causal cutoff: classify all refreshed text, then tolerate at most
    one second of later PCM. This trades semantic lag for reachable interruption
    during continuous speech; later qualifiers remain possible. The host keeps
    all input for the eventual answer, and must never label this path current.
    """
    def __init__(self, *, allow_control=False, min_audio_ms=320,
                 interval_ms=320, stability_ms=160, max_partial_probes=3,
                 max_audio_ms=MAX_OBSERVATION_SECONDS * 1000,
                 max_result_age_ms=2000, max_causal_lag_ms=1000, clock=time.monotonic):
        integers = (min_audio_ms, interval_ms, stability_ms, max_partial_probes,
                    max_audio_ms, max_result_age_ms, max_causal_lag_ms)
        if (type(allow_control) is not bool
                or any(type(v) is not int or v <= 0 for v in integers)
                or not stability_ms <= min_audio_ms <= max_audio_ms
                or max_audio_ms > MAX_OBSERVATION_SECONDS * 1000
                or max_causal_lag_ms > 1000):
            raise ValueError("Invalid bounded listener configuration.")
        self.allow_control = allow_control
        self.min_audio_ms, self.interval_ms = min_audio_ms, interval_ms
        self.stability_ms, self.max_partial_probes = stability_ms, max_partial_probes
        self.max_audio_ms, self.max_result_age_ms = max_audio_ms, max_result_age_ms
        self.max_causal_lag_ms = max_causal_lag_ms
        self.clock = clock
        self._epoch = self._revision = 0
        self._active = False
        self._inflight = self._refresh = None
        self._revalidation_reserved = False
        self._pcm = bytearray()
        self._reset_input()

    def _reset_input(self):
        self._pcm.clear()
        self._samples = self._last_requested = self._partial_probes = 0
        self._final = self._final_requested = self._applied = self.exhausted = False
        self._refresh = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def retained_bytes(self) -> int:
        """Buffered bytes only; caller-owned snapshots have their own lifetime."""
        return len(self._pcm)

    @property
    def inflight(self) -> ListeningSnapshot | None:
        return self._inflight

    def begin(self, response_id: str, utterance_id: str, *, sample_rate=24000, start_ms=0):
        if any(not isinstance(v, str) or not v or len(v) > 256
               for v in (response_id, utterance_id)):
            raise ValueError("Response and utterance IDs must be nonempty bounded text.")
        if (type(sample_rate) is not int or sample_rate not in (16000, 24000)
                or type(start_ms) is not int or start_ms < 0):
            raise ValueError("Use PCM16 at 16 or 24 kHz with a nonnegative start time.")
        self._epoch += 1
        self._reset_input()
        self.response_id, self.utterance_id = response_id, utterance_id
        self.sample_rate, self.start_ms = sample_rate, start_ms
        self._active = True

    def append(self, pcm: bytes | bytearray | memoryview) -> bool:
        if not self._active or self._final:
            raise ValueError("Begin an unfinished utterance before appending audio.")
        if not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise TypeError("Append mono little-endian PCM16 bytes.")
        view = memoryview(pcm).cast("B")
        if len(view) % 2:
            raise ValueError("PCM16 samples require an even byte count.")
        self._samples += len(view) // 2
        limit = self.max_audio_ms * self.sample_rate // 1000 * 2
        self._pcm.extend(view[:max(0, limit - len(self._pcm))])
        self.exhausted = self._samples * 1000 > self.max_audio_ms * self.sample_rate
        # This only disables the disposable observer. The caller must retain all
        # original user PCM for the ordinary response path. No acoustic yield.
        return not self.exhausted

    def finish_input(self) -> ListeningSnapshot | None:
        """Invalidate partial control; return its ticket for owner cancellation.

        Its reservation remains until real retirement. Then a final snapshot can
        be requested even if the partial-probe budget was already consumed.
        """
        if not self._active:
            raise ValueError("No active listener utterance.")
        if not self._final:
            self._epoch += 1
            self._refresh = None
            self._final = True
        return self._inflight

    def _snapshot(self, revision: int) -> ListeningSnapshot:
        return ListeningSnapshot(self.response_id, self.utterance_id, self._epoch,
                                 revision, bytes(self._pcm), self.sample_rate,
                                 self.start_ms, self._final, self.clock())

    def request(self) -> ListeningSnapshot | None:
        if (not self._active or self._inflight is not None or self.exhausted
                or self._applied or not self._samples):
            return None
        if self._final:
            if self._final_requested:
                return None
            self._final_requested = True
        else:
            if (self._samples * 1000 < self.min_audio_ms * self.sample_rate
                    or self._partial_probes >= self.max_partial_probes
                    or (self._last_requested and
                        (self._samples - self._last_requested) * 1000
                        < self.interval_ms * self.sample_rate)):
                return None
            self._partial_probes += 1
        self._revision += 1
        self._last_requested = self._samples
        self._inflight = self._snapshot(self._revision)
        return self._inflight

    def refresh(self, ticket: ListeningSnapshot) -> ListeningSnapshot | None:
        if (ticket is not self._inflight or not self._active
                or ticket.epoch != self._epoch or self.exhausted):
            return None
        self._refresh = self._snapshot(ticket.revision)
        return self._refresh

    def reserve_revalidation(self, ticket: ListeningSnapshot) -> bool:
        """Charge the optional second classification to the same brain budget."""
        if (ticket is not self._inflight or not self._active or ticket.epoch != self._epoch
                or self.exhausted or self._revalidation_reserved
                or self._partial_probes >= self.max_partial_probes):
            return False
        self._partial_probes += 1
        self._revalidation_reserved = True
        return True

    def retire(self, ticket: ListeningSnapshot):
        """Call once the real owner has relinquished a cancelled/completed job."""
        if ticket is self._inflight:
            self._inflight = self._refresh = None
            self._revalidation_reserved = False

    def cancel(self):
        self._epoch += 1
        self._active = False
        self._reset_input()
        # Do not clear _inflight: only actual owner retirement frees the slot.

    def complete(self, ticket: ListeningSnapshot, evidence: PrefixEvidence,
                 decision: ListeningDecision, *, recheck: PrefixEvidence | None = None,
                 revalidated: ListeningDecision | None = None,
                 semantic_skipped: bool = False) -> ListeningResult:
        """Return a side-effect-free floor recommendation and retire this ticket.

        ``apply`` is never a playback pause. The adapter may atomically relinquish
        the matching response's floor, preserving heard prefix + unfinished draft
        and all user audio. It must not start a reply from a partial observation.
        ``backchannel``/``wait`` never pause, clear, or consume the user turn.
        ``revalidated`` is a second semantic decision about the complete
        ``recheck`` transcript, not permission to ignore a recognized suffix.
        It still requires the initial positive's acoustic stability; the second
        observation may contain a newly growing word. No third validation loop
        is implied, and additional unclassified PCM is bounded explicitly.
        A skipped empty/unstable acoustic preflight may refund its reserved
        semantic probe. Every acoustic request still advances the 320ms minimum
        interval, and the full six-second PCM budget continues to bound work.
        """
        reason = "eligible"
        latest = recheck or evidence
        selected = revalidated or decision
        age_ms = (self.clock() - (latest.snapshot.captured_at if revalidated else ticket.captured_at)) * 1000
        age_limit = self.max_causal_lag_ms if revalidated else self.max_result_age_ms
        if (ticket is not self._inflight or not self._active or ticket.epoch != self._epoch):
            reason = "stale_ticket"
        elif evidence.snapshot is not ticket:
            reason = "wrong_evidence"
        elif semantic_skipped and (
                type(semantic_skipped) is not bool or recheck is not None or revalidated is not None
                or self._revalidation_reserved
                or decision != ListeningDecision("wait")
                or (evidence.has_words and (ticket.final or evidence.stable(self.stability_ms)))):
            reason = "invalid_semantic_skip"
        elif semantic_skipped:
            if not ticket.final:
                self._partial_probes -= 1
            reason = "semantic_skipped"
        elif self.exhausted:
            reason = "audio_budget"
        elif recheck is not None and recheck.snapshot is not self._refresh:
            reason = "wrong_recheck"
        elif revalidated is not None and (recheck is None or decision.intent != "take_floor"
                                          or decision.addressed_to_assistant is not True
                                          or not decision.sufficient_evidence):
            reason = "invalid_revalidation"
        elif not math.isfinite(age_ms) or not 0 <= age_ms <= age_limit:
            reason = "expired"
        elif revalidated is None and not latest.same_transcript(evidence):
            reason = "changed_transcript"
        elif revalidated is None and latest.snapshot.samples != self._samples:
            reason = "newer_audio"
        elif revalidated is not None and (self._samples - latest.snapshot.samples) * 1000 > (
                self.max_causal_lag_ms * self.sample_rate):
            reason = "causal_audio_lag"
        elif not _words(evidence.text) or not _words(latest.text):
            reason = "empty_evidence"
        elif not (evidence.snapshot.final or evidence.stable(self.stability_ms)):
            reason = "unstable_evidence"
        elif selected.intent != "take_floor":
            reason = selected.intent
        elif selected.addressed_to_assistant is not True:
            reason = "uncertain_addressee"
        elif not selected.sufficient_evidence:
            reason = "insufficient_evidence"
        eligible = reason == "eligible"
        apply = eligible and self.allow_control
        if apply:
            self._applied = True
            if revalidated is not None:
                reason = "causal_cutoff"
        result = ListeningResult(ticket.response_id, ticket.utterance_id, ticket.epoch,
                                 ticket.revision, latest.snapshot.samples, selected.action,
                                 selected.intent, reason if not eligible or apply else "observation_only",
                                 eligible, apply)
        self.retire(ticket)
        return result
