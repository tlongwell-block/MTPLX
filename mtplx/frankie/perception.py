"""Experimental semantic observations on Frankie's existing inference owner.

There is no extra ear/brain instance or inference thread. Acoustic preparation
and short text probes use the same bounded scheduler as concurrent HTTP work.
"""

import asyncio
import time
from dataclasses import replace

import numpy as np

from .floor import floor_messages, parse_floor_decision
from .interaction import (
    MAX_OBSERVATION_SECONDS,
    ListenerObservation,
    parse_compact_decision,
    semantic_messages,
)
from .listening import ListeningDecision, PrefixEvidence


class RealtimeListener:
    def __init__(self, session, service):
        self.session, self.service = session, service
        self.job = None
        self.revision = 0
        self._generation = 0

    def cancel(self):
        self._generation += 1
        if self.job is not None:
            self.job.cancelled.set()
            self.job.ready.set()

    async def _admit(self):
        self.cancel()
        generation = self._generation
        # Only the inference owner retires work/caches. Cancellation alone does
        # not free admission, including when a final turn supersedes a prefix.
        deadline = time.monotonic() + 3
        while any(j.internal for j in tuple(self.service.jobs)):
            if self._generation != generation:
                raise RuntimeError("Listener admission was cancelled.")
            if time.monotonic() >= deadline or self.session.closed:
                raise TimeoutError("Listener admission deadline.")
            await asyncio.sleep(0.01)
        if self.session.closed or self._generation != generation:
            raise RuntimeError("Listener admission was cancelled.")
        self.revision += 1

    @staticmethod
    def _data():
        return {"messages": [], "max_tokens": 6, "temperature": 0, "seed": 0,
                "presence_penalty": 0, "frequency_penalty": 0,
                "thinking": "off", "enable_thinking": False, "tools": []}

    async def _run(self, data, prepare, *, prepare_only=False):
        started = time.monotonic()
        job = (self.service.submit_prepare(prepare) if prepare_only else
               self.service.submit(data, True, internal=True, prepare=prepare))
        self.job = job
        stats = {}
        try:
            async with asyncio.timeout(5):
                while True:
                    event = await job.receive()
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    if "listener_prepare_seconds" in event:
                        stats["ear_ms"] = event["listener_prepare_seconds"] * 1000
                    if "finish_reason" in event or "prepared" in event:
                        stats.update(event.get("usage", {}),
                                     elapsed_ms=(time.monotonic() - started) * 1000)
                        return event, stats
        finally:
            job.cancelled.set()
            job.ready.set()
            # Return only after the actual owner has relinquished the job. A
            # bounded timeout leaves self.job pointing to unretired work, so a
            # subsequent admission cannot treat cancellation as free capacity.
            async with asyncio.timeout(3):
                while job in self.service.jobs:
                    await asyncio.sleep(0.01)
            if self.job is job:
                self.job = None

    async def classify(self, item, run, *, fragments=None):
        await self._admit()
        parts = [part for fragment in (fragments or (item,))
                 for part in fragment["content"] if part["type"] == "input_audio"]
        duration_ms = sum(len(part["_pcm"]) * 1000 / part.get("_rate", 24000)
                          for part in parts)
        streaming = self.session.settings.get("streaming_listener", "off") in {"observe", "semantic"}
        limit = 90 if streaming else MAX_OBSERVATION_SECONDS
        if duration_ms > limit * 1000:
            raise ValueError("Final listener input exceeds 90 seconds." if streaming else
                             "Semantic observation exceeds the six-second acoustic budget.")
        heard = " ".join(c["text"] for c in run.chunks if c["end_ms"] <= run.played_ms)
        observation = ListenerObservation(
            utterance_id=item["id"], revision=self.revision,
            # The heard prefix is sampled now, after endpointing. Do not label
            # this snapshot as if it were available at the earlier speech end.
            observed_ms=self.session.clock_ms,
            user_text="", assistant_heard_text=heard,
            assistant_speaking=True, user_speaking=False, is_final=True,
            speech_ms=round(duration_ms),
            pending_work=any(t.status == "running" for t in self.session.task_ledger.tasks.values()),
            assistant_name=self.session.settings.get("assistant_name", "Frankie"),
        )
        data = self._data()
        prepared = {}

        def prepare(job):
            for part in parts:
                if job.cancelled.is_set():
                    raise RuntimeError("Listener observation was cancelled.")
                if "_rows" not in part or "_transcript" not in part:
                    # Publish completed ear work on its existing model owner.
                    # A superseding probe waits for this job to retire before
                    # reusing these immutable features, even after cancellation.
                    part["_rows"], part["_transcript"] = self.session.engine.audio.hear(
                        part["_pcm"], part.get("_rate", 24000)
                    )
            observed = replace(observation, user_text=" ".join(
                part["_transcript"].strip() for part in parts if part["_transcript"].strip()))
            prepared["observation"] = observed
            job.data["messages"] = (floor_messages(observed) if streaming else
                                    semantic_messages(observed, compact=True))

        event, stats = await self._run(data, prepare)
        observed = prepared["observation"]
        decision = (parse_floor_decision(event["message"]["content"]) if streaming else
                    parse_compact_decision(event["message"]["content"], observed))
        return decision, observed, stats

    def _hear_prefix(self, snapshot, job, *, compare):
        if (snapshot.sample_rate not in (16000, 24000) or not snapshot.pcm
                or len(snapshot.pcm) % 2
                or snapshot.samples > MAX_OBSERVATION_SECONDS * snapshot.sample_rate):
            raise ValueError("Invalid or oversized disposable listener PCM.")
        pcm = np.frombuffer(snapshot.pcm, dtype="<i2").astype(np.float32) / 32768
        previous_samples = snapshot.samples - snapshot.sample_rate * 160 // 1000
        previous_text = None
        if compare and previous_samples > 0:
            if job.cancelled.is_set():
                raise RuntimeError("Listener observation was cancelled.")
            _, previous_text = self.session.engine.audio.hear(pcm[:previous_samples], snapshot.sample_rate)
        else:
            previous_samples = None
        if job.cancelled.is_set():
            raise RuntimeError("Listener observation was cancelled.")
        _, text = self.session.engine.audio.hear(pcm, snapshot.sample_rate)
        # Both temporary feature arrays die here; no conversation part is read
        # or mutated, and no partial rows/transcript can poison final input.
        return PrefixEvidence(snapshot, text, previous_text, previous_samples)

    async def classify_prefix(self, snapshot, *, heard_text, pending_work, assistant_name,
                              require_stable=True):
        if type(require_stable) is not bool:
            raise ValueError("Prefix stability policy must be boolean.")
        started = time.monotonic()
        await self._admit()
        generation = self._generation
        prepared = {}

        def prepare(job):
            evidence = self._hear_prefix(snapshot, job, compare=require_stable)
            # Ear screening and an eligible brain probe share one admission.
            # Check supersession before warming/tokenizing on that same owner.
            if job.cancelled.is_set() or self.session.closed or generation != self._generation:
                raise RuntimeError("Listener observation was cancelled after acoustic preflight.")
            prepared["evidence"] = evidence
            if not evidence.has_words or (require_stable and not snapshot.final and not evidence.stable(160)):
                return evidence
            observation = ListenerObservation(
                utterance_id=snapshot.utterance_id, revision=snapshot.revision,
                observed_ms=round(snapshot.observed_ms), user_text=evidence.text,
                assistant_heard_text=heard_text, assistant_speaking=True,
                user_speaking=not snapshot.final, is_final=snapshot.final,
                transcript_stable=evidence.stable(160),
                speech_ms=round(snapshot.samples * 1000 / snapshot.sample_rate),
                pending_work=pending_work, assistant_name=assistant_name,
            )
            job.data["messages"] = floor_messages(observation)

        event, stats = await self._run(self._data(), prepare)
        evidence = prepared["evidence"]
        if self.session.closed or generation != self._generation:
            raise RuntimeError("Listener observation was cancelled after acoustic preflight.")
        skipped = "prepared" in event
        stats.update(semantic_skipped=skipped, elapsed_ms=(time.monotonic() - started) * 1000)
        if skipped:
            stats["semantic_skip_reason"] = "empty" if not evidence.has_words else "unstable"
            return ListeningDecision("wait"), evidence, stats
        decision = parse_floor_decision(event["message"]["content"])
        return decision, evidence, stats

    async def recheck_prefix(self, snapshot):
        await self._admit()
        event, _ = await self._run({}, lambda job: self._hear_prefix(snapshot, job, compare=False),
                                   prepare_only=True)
        return event["prepared"]
