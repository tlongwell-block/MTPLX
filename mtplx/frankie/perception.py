"""Experimental semantic observations on Frankie's existing inference owner.

There is no extra ear/brain instance or inference thread. Acoustic preparation
and short text probes use the same bounded scheduler as concurrent HTTP work.
"""

import asyncio
import time
from dataclasses import replace

from .interaction import (
    MAX_OBSERVATION_SECONDS,
    ListenerObservation,
    parse_compact_decision,
    semantic_messages,
)


class RealtimeListener:
    def __init__(self, session, service):
        self.session, self.service = session, service
        self.job = None
        self.revision = 0

    def cancel(self):
        if self.job is not None:
            self.job.cancelled.set()
            self.job.ready.set()

    async def classify(self, item, run, *, fragments=None):
        self.cancel()
        # Cancelled work is released by the inference owner, never by the
        # websocket thread while a model slice still owns its cache.
        deadline = time.monotonic() + 3
        while any(j.internal for j in tuple(self.service.jobs)):
            if time.monotonic() >= deadline or self.session.closed:
                raise TimeoutError("Listener admission deadline.")
            await asyncio.sleep(0.01)
        self.revision += 1
        parts = [part for fragment in (fragments or (item,))
                 for part in fragment["content"] if part["type"] == "input_audio"]
        duration_ms = sum(len(part["_pcm"]) * 1000 / part.get("_rate", 24000)
                          for part in parts)
        if duration_ms > MAX_OBSERVATION_SECONDS * 1000:
            raise ValueError("Semantic observation exceeds the six-second acoustic budget.")
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
        data = {"messages": [], "max_tokens": 6, "temperature": 0, "seed": 0,
                "thinking": "off", "enable_thinking": False, "tools": []}
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
            job.data["messages"] = semantic_messages(observed, compact=True)

        started = time.monotonic()
        job = self.service.submit(data, True, internal=True, prepare=prepare)
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
                    if "finish_reason" in event:
                        observed = prepared["observation"]
                        decision = parse_compact_decision(event["message"]["content"], observed)
                        stats.update(event.get("usage", {}),
                                     elapsed_ms=(time.monotonic() - started) * 1000)
                        return decision, observed, stats
        finally:
            job.cancelled.set()
            job.ready.set()
            if self.job is job:
                self.job = None
