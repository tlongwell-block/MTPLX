"""Realtime conversation ownership, VAD, cancellation and heard-prefix history."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from threading import Event

import numpy as np

from .audio import resample
from .interaction import MAX_OBSERVATION_SECONDS
from .tasks import (
    BACKGROUND_TASK_INSTRUCTIONS,
    TaskLedger,
    project_history,
    task_notice,
)


def identifier(prefix):
    return prefix + "_" + uuid.uuid4().hex[:20]


def public(item):
    if isinstance(item, dict):
        return {k: public(v) for k, v in item.items() if not k.startswith("_")}
    if isinstance(item, list):
        return [public(v) for v in item]
    return item


@dataclass
class Response:
    id: str = field(default_factory=lambda: identifier("resp"))
    item_id: str = field(default_factory=lambda: identifier("item"))
    abort: Event = field(default_factory=Event)
    ready: Event = field(default_factory=Event)
    input: dict | None = None
    visible: bool = False
    committed_at: float = 0.0
    text: str = ""
    chunks: list = field(default_factory=list)
    item: dict | None = None
    done: bool = False
    settings: dict = field(default_factory=dict)
    transcripts: list = field(default_factory=list)
    speech_end_ms: int = 0
    played_ms: int = 0
    merged: bool = False
    emitted_ms: float = 0.0
    playback_finished: bool = False
    first_audio_at: float = 0.0
    task_results: set = field(default_factory=set)


class Session:
    def __init__(self, engine, executor, websocket):
        self.engine, self.executor, self.ws = engine, executor, websocket
        self.loop = asyncio.get_running_loop()
        self.id = identifier("sess")
        self.items = []
        self.latest_context = (0, "")
        self.capture_context = (0, "")
        self.capture_task_results = ()
        self.current = None
        self.spec = None
        self.closed = False
        self.tasks = set()
        self.task_ledger = TaskLedger()
        self.input_revision = 0
        self.user_revision = 0
        self.response_revision = -1
        self.queued_task_response = False
        self.unhandled_task_results = set()
        self.playback_runs = {}
        self.listener = None
        self.overlap_run = None
        self.overlap_prefix_ms = 0
        self.semantic_pending = 0
        self.semantic_epoch = 0
        self.speech_paused = False
        self.playback_wake = None
        self.outgoing = asyncio.Queue(maxsize=2048)
        self.settings = {
            "instructions": "You are Frankie. Be helpful and concise. Use natural spoken sentences.",
            "output_modalities": ["audio"],
            "max_output_tokens": 512,
            "thinking": "off",
            "context": 131072,
            "tools": [],
            "input_rate": 24000,
            "background_tasks": False,
            "interruption_policy": "vad",
            "assistant_name": "Frankie",
            "playback_feedback": False,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 320,
                "prefix_padding_ms": 320,
                "create_response": True,
                "interrupt_response": True,
            },
        }
        self.vad = engine.audio.make_vad()
        make_turn = getattr(engine.audio, "make_turn", None)
        self.turn = make_turn() if make_turn else None
        self.tail = np.empty(0, dtype=np.float32)
        self.system_tail = np.empty(0, dtype=np.float32)
        self.pre = deque(maxlen=10)
        self.frames = []
        self.silence = 0
        self.voice_run = 0
        self.listening = False
        self.manual = []
        self.manual_size = 0
        self.clock_ms = 0
        self.received_ms = 0.0
        self.speech_id = None
        self.speech_start_ms = 0
        self.last_speech_ms = 0
        self.metrics = {
            "input_frames": 0,
            "input_frames_during_output": 0,
            "speculations": 0,
            "speculation_aborts": 0,
            "barge_ins": 0,
        }

    def event(self, type, **data):
        if self.closed:
            return
        try:
            self.outgoing.put_nowait(
                {"type": type, "event_id": identifier("evt"), **data}
            )
        except asyncio.QueueFull:
            if self.current:
                self.current.abort.set()
            self.closed = True
            asyncio.create_task(
                self.ws.close(code=1013, reason="Client is not consuming audio.")
            )

    async def send_events(self):
        while True:
            event = await self.outgoing.get()
            await self.ws.send_json(event)

    def info(self):
        return {
            "id": self.id,
            "type": "realtime",
            "model": "Frankie",
            "object": "realtime.session",
            "frankie": {
                "input_context": True,
                "background_tasks": self.settings["background_tasks"],
                "playback_feedback": True,
                "interruption_policy": self.settings["interruption_policy"],
                "semantic_listener_available": self.listener is not None,
                "assistant_name": self.settings["assistant_name"],
            },
            "output_modalities": self.settings["output_modalities"],
            "instructions": self.settings["instructions"],
            "max_output_tokens": self.settings["max_output_tokens"],
            "thinking": self.settings["thinking"],
            "reasoning": {
                "effort": self.settings["thinking"]
                if self.settings["thinking"] != "off"
                else "none"
            },
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.settings["input_rate"],
                    },
                    "turn_detection": self.settings["turn_detection"],
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            },
            "tools": self.settings["tools"],
        }

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def show(self, run):
        if run.visible or run.abort.is_set():
            return
        if run.input is not None and not any(
            x["id"] == run.input["id"] for x in self.items
        ):
            self.items.append(run.input)
            self.accept_input()
            self.event(
                "input_audio_buffer.committed",
                item_id=run.input["id"],
                previous_item_id=None,
            )
            self.event(
                "conversation.item.created",
                item=public(run.input),
                previous_item_id=None,
            )
        run.visible = True
        self.response_revision = self.user_revision
        for call_id in run.task_results:
            task = self.task_ledger.tasks.get(call_id)
            if task is not None:
                task.consumed = True
        self.unhandled_task_results.difference_update(run.task_results)
        if not self.unhandled_task_results:
            self.queued_task_response = False
        self.playback_runs[run.item_id] = run
        # Keep a bounded feedback window. Old runs can no longer mutate history.
        while len(self.playback_runs) > 16:
            del self.playback_runs[next(iter(self.playback_runs))]
        run.committed_at = time.monotonic()
        for value in run.transcripts:
            self.transcribed(*value)
        run.transcripts.clear()
        audio = "audio" in run.settings["output_modalities"]
        run.item = {
            "id": run.item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [
                {"type": "output_audio", "transcript": ""}
                if audio
                else {"type": "output_text", "text": ""}
            ],
        }
        self.items.append(run.item)
        self.event(
            "response.created",
            response={"id": run.id, "status": "in_progress", "output": []},
        )
        self.event(
            "response.output_item.added",
            response_id=run.id,
            output_index=0,
            item=public(run.item),
        )
        self.event(
            "conversation.item.created", item=public(run.item), previous_item_id=None
        )
        self.event(
            "response.content_part.added",
            response_id=run.id,
            item_id=run.item_id,
            output_index=0,
            content_index=0,
            part=public(run.item["content"][0]),
        )
        run.ready.set()

    def emit(self, run, kind, value):
        # Transcription must not stall speculative brain prefill. Hold its
        # event on the event-loop side until the same utterance is committed.
        if kind != "input_transcript":
            while not run.ready.wait(0.01):
                if run.abort.is_set() or self.closed:
                    return
        if run.abort.is_set() or self.closed:
            return
        if kind == "audio":
            value = base64.b64encode(
                np.clip(value * 32767, -32768, 32767).astype("<i2").tobytes()
            ).decode()
        self.loop.call_soon_threadsafe(self.publish, run, kind, value)

    def publish(self, run, kind, value):
        if run.abort.is_set() or self.closed:
            return
        if kind == "input_transcript":
            if run.visible:
                self.transcribed(*value)
            else:
                run.transcripts.append(value)
            return
        common = {
            "response_id": run.id,
            "item_id": run.item_id,
            "output_index": 0,
            "content_index": 0,
        }
        if kind == "audio":
            if not run.first_audio_at:
                run.first_audio_at = time.monotonic()
            raw_size = len(value) // 4 * 3 - (2 if value.endswith("==") else value.endswith("="))
            run.emitted_ms += raw_size * 1000 / (24000 * 2)
            self.event("response.output_audio.delta", delta=value, **common)
        elif kind == "text":
            run.text += value
            part = run.item["content"][0]
            part["transcript" if part["type"] == "output_audio" else "text"] = run.text
            self.event(
                "response.output_audio_transcript.delta"
                if part["type"] == "output_audio"
                else "response.output_text.delta",
                delta=value,
                **common,
            )
        elif kind == "chunk":
            run.chunks.append(value)

    def transcribed(self, item, index, transcript):
        if self.closed or not any(i is item for i in self.items):
            return
        part = item["content"][index]
        if part.get("_transcript_sent"):
            return
        part["transcript"] = transcript
        part["_transcript_sent"] = True
        self.event(
            "conversation.item.input_audio_transcription.completed",
            item_id=item["id"],
            content_index=index,
            transcript=transcript,
        )

    async def transcribe_item(self, item):
        try:
            transcripts = await self.loop.run_in_executor(
                self.executor, lambda: self.engine.prepare_audio(item)
            )
            for value in transcripts:
                self.transcribed(*value)
        except Exception as exc:  # noqa: BLE001 — report model/backend failures to the client.
            if not self.closed and any(i is item for i in self.items):
                self.event(
                    "conversation.item.input_audio_transcription.failed",
                    item_id=item["id"],
                    content_index=next(
                        i
                        for i, p in enumerate(item["content"])
                        if p["type"] == "input_audio"
                    ),
                    error={"type": "transcription_error", "message": str(exc)},
                )

    def start(self, input_item=None, *, tentative=False, deferred_task_results=None):
        if (
            self.current is not None
            and not self.current.done
            and not self.current.abort.is_set()
        ):
            raise ValueError("A response is already active.")
        # A completed generation can still have unheard audio. Explicit new
        # user/manual responses replace that tail before taking their snapshot;
        # result-only followups wait for drain in maybe_start_task_response.
        audible = self.audible_response()
        if audible is not None:
            self.rollback_unheard(audible)
        if input_item is None and self.user_revision > self.response_revision:
            latest_user = next((item for item in reversed(self.items)
                                if item.get("role") == "user"), None)
            if latest_user is not None and "_task_results_at_capture" in latest_user:
                input_item = latest_user
        run = Response(
            input=input_item,
            settings=copy.deepcopy(self.settings),
            speech_end_ms=(input_item or {}).get("_speech_end_ms", self.last_speech_ms),
        )
        self.current = run
        history = list(self.items) + (
            [input_item]
            if input_item is not None and input_item not in self.items
            else []
        )
        if self.settings["background_tasks"]:
            # An already captured audio turn gets its answer first. A
            # result queued while that turn was being classified belongs to a
            # later response, not merely to a snapshot that might ignore it.
            # Keep authoritative items and their arrival order unchanged.
            if deferred_task_results is None:
                captured = (input_item or {}).get("_task_results_at_capture")
                deferred_task_results = (self.unhandled_task_results - set(captured)
                                         if captured is not None and self.queued_task_response else ())
            deferred = self.unhandled_task_results.intersection(deferred_task_results)
            if deferred:
                history = [item for item in history
                           if not (item["type"] == "function_call_output"
                                   and item["call_id"] in deferred)]
            history = project_history(history)
            run.settings["instructions"] += "\n\n" + BACKGROUND_TASK_INSTRUCTIONS
            run.task_results = self.unhandled_task_results - deferred
        if tentative:
            self.spec = run
            self.metrics["speculations"] += 1
            self.event("frankie.speculation.started", response_id=run.id)
        else:
            self.show(run)
        self.spawn(self.generate(run, history))
        return run

    async def generate(self, run, history):
        result = None
        error = None
        try:
            result = await self.loop.run_in_executor(
                self.executor,
                lambda: self.engine.respond(
                    history,
                    run.settings,
                    lambda k, v: self.emit(run, k, v),
                    run.abort,
                    session_id=self.id,
                ),
            )
        except Exception as exc:  # noqa: BLE001 — keep backend failures within this response.
            if not run.abort.is_set():
                import traceback

                traceback.print_exc()
                error = str(exc)
        run.done = True
        if self.closed:
            return
        if not run.visible:
            return
        status = (
            "cancelled" if run.abort.is_set() else ("failed" if error else "completed")
        )
        output = []
        if run.item is not None:
            run.item["status"] = "completed" if status == "completed" else "incomplete"
            output.append(public(run.item))
        if result and status == "completed":
            from .thinking import public_tool_calls

            calls = public_tool_calls(
                result["raw_text"],
                self.engine.tokenizer,
                run.settings["tools"],
                starts_in_thinking=run.settings.get("thinking", "off") != "off",
            )
            for call in calls:
                function = call["function"]
                item = {
                    "id": identifier("item"),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call.get("id", identifier("call")),
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
                if self.settings["background_tasks"]:
                    # Parser-generated identifiers must never collide with a
                    # historical invocation, including evicted ledger entries.
                    while any(i.get("call_id") == item["call_id"] for i in self.items):
                        item["call_id"] = identifier("call")
                    try:
                        task = self.task_ledger.register(item, self.input_revision, run.id)
                    except ValueError as exc:
                        error, status = str(exc), "failed"
                        break
                    self.event("frankie.task.updated", task=task.public())
                self.items.append(item)
                output.append(item)
                self.event(
                    "response.output_item.added",
                    response_id=run.id,
                    output_index=len(output) - 1,
                    item=item,
                )
                self.event(
                    "response.function_call_arguments.done",
                    response_id=run.id,
                    item_id=item["id"],
                    call_id=item["call_id"],
                    name=item["name"],
                    arguments=item["arguments"],
                    output_index=len(output) - 1,
                )
                self.event(
                    "response.output_item.done",
                    response_id=run.id,
                    output_index=len(output) - 1,
                    item=item,
                )
            if result["finish_reason"] == "length":
                status = "incomplete"
            self.event(
                "frankie.metrics",
                response_id=run.id,
                metrics={
                    **self.metrics,
                    **result["stats"],
                    "audio_seconds": result["audio_seconds"],
                    "wall_seconds": result["seconds"],
                },
            )
        common = {
            "response_id": run.id,
            "item_id": run.item_id,
            "output_index": 0,
            "content_index": 0,
        }
        if "audio" in run.settings["output_modalities"]:
            self.event("response.output_audio.done", **common)
            self.event(
                "response.output_audio_transcript.done",
                transcript=run.item["content"][0].get("transcript", ""),
                **common,
            )
        else:
            self.event(
                "response.output_text.done",
                text=run.item["content"][0].get("text", ""),
                **common,
            )
        self.event(
            "response.content_part.done",
            part=public(run.item["content"][0]),
            **common,
        )
        self.event(
            "response.output_item.done",
            response_id=run.id,
            output_index=0,
            item=public(run.item),
        )
        response = {"id": run.id, "status": status, "output": output}
        if status == "incomplete":
            response["status_details"] = {
                "type": "incomplete",
                "reason": "max_output_tokens",
            }
        if error:
            response["status_details"] = {"type": "failed", "error": {"message": error}}
        self.event("response.done", response=response)
        self.maybe_start_task_response()

    def maybe_start_task_response(self):
        if (
            self.queued_task_response
            and self.unhandled_task_results
            and not self.closed
            and not self.listening
            and not self.manual_size
            and not self.semantic_pending
            and not self.speech_paused
            and (not self.user_revision or self.response_revision == self.user_revision)
            and (self.current is None or self.current.done)
        ):
            audible = self.audible_response()
            if audible is not None:
                if not self.settings["playback_feedback"]:
                    if self.playback_wake is not None:
                        self.playback_wake.cancel()
                    delay = max(0.01, audible.first_audio_at
                                + audible.emitted_ms / 1000 + 0.5 - time.monotonic())
                    self.playback_wake = self.loop.call_later(delay, self.playback_deadline)
                return
            self.start()

    def playback_deadline(self):
        self.playback_wake = None
        self.maybe_start_task_response()

    def invalidate_semantics(self):
        self.semantic_epoch += 1
        if self.listener is not None:
            self.listener.cancel()

    def accept_input(self, *, semantic_overlap=False):
        """Prioritize new input without losing an already requested result reply."""
        self.input_revision += 1
        self.user_revision += 1
        self.speech_paused = False
        if not semantic_overlap:
            self.invalidate_semantics()

    def cancel(self):
        run = self.current
        if run is not None and not run.done:
            run.abort.set()
            run.ready.set()
        return run

    def merge_resumed(self, run):
        if (
            not run.input
            or not run.visible
            or run.merged
            or (run.done and not run.abort.is_set())
            or time.monotonic() - run.committed_at >= 0.7
            or not 0 <= self.speech_start_ms - run.speech_end_ms < 700
            or run.played_ms >= 700
        ):
            return
        index = next(
            (i for i, item in enumerate(self.items) if item is run.input), None
        )
        if index is None or any(
            item["type"] in {"function_call", "function_call_output"}
            for item in self.items[index:]
        ):
            return
        run.merged = True
        self.frames.insert(
            0,
            next(p["_pcm"] for p in run.input["content"] if p["type"] == "input_audio"),
        )
        self.capture_context = run.input.get("_capture_context", (0, ""))
        removed = {run.input["id"], run.item_id}
        self.items = [item for item in self.items if item["id"] not in removed]
        for item_id in removed:
            self.event("conversation.item.deleted", item_id=item_id)
        self.event("frankie.input.merged", response_id=run.id)

    def discard_spec(self):
        if self.spec is None:
            return
        self.spec.abort.set()
        self.spec.ready.set()
        self.metrics["speculation_aborts"] += 1
        self.event("frankie.speculation.aborted", response_id=self.spec.id)
        self.spec = None

    def audio_item(self, pcm, *, item_id=None, speech_end_ms=None):
        item = {
            "id": item_id or identifier("item"),
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "_pcm": pcm,
                    "_rate": self.settings["input_rate"],
                }
            ],
        }
        self.attach_context(item, self.capture_context)
        item["_capture_context"] = self.capture_context
        item["_task_results_at_capture"] = self.capture_task_results
        if speech_end_ms is not None:
            item["_speech_start_ms"] = self.speech_start_ms
            item["_speech_end_ms"] = speech_end_ms
        return item

    def next_context(self):
        revision, text = self.latest_context
        if any(item.get("_context_revision") == revision for item in self.items):
            return (0, "")
        return revision, text

    @staticmethod
    def attach_context(item, context):
        revision, text = context
        if text:
            item["content"].insert(0, {"type": "input_text", "text": text})
            item["_context_revision"] = revision

    def reset_detectors(self):
        self.vad.reset()
        self.tail = np.empty(0, dtype=np.float32)
        self.system_tail = np.empty(0, dtype=np.float32)
        self.pre.clear()
        if self.turn is not None:
            self.turn.reset(self.received_ms)

    def audible_response(self):
        run = self.current
        if (run is None or not run.visible or run.abort.is_set()
                or run.playback_finished or run.emitted_ms <= 0):
            return None
        if not run.done:
            return run
        if self.settings["playback_feedback"]:
            return run
        # Clients without playback feedback must not leave a completed response
        # looking audible forever. The direct demo reports exact drain instead.
        if run.first_audio_at and time.monotonic() < run.first_audio_at + run.emitted_ms / 1000 + 0.5:
            return run
        return None

    def preceding_overlaps(self, run, start_ms, *, before=None):
        """Recover nearby unfinished fragments, without carrying older turns."""
        end = next((i for i, item in enumerate(self.items) if item is before), len(self.items))
        fragments = []
        for item in reversed(self.items[:end]):
            if item["type"] in {"function_call", "function_call_output"}:
                continue  # A background result does not end the user's phrase.
            if (item.get("_listener_response") != run.id
                    or item.get("_listener_consumed")
                    or not 0 <= start_ms - item.get("_speech_end_ms", -2000) <= 1500):
                break
            fragments.append(item)
            start_ms = item["_speech_start_ms"]
        return tuple(reversed(fragments))

    @staticmethod
    def fragment_audio_ms(items):
        return sum(len(part["_pcm"]) * 1000 / part.get("_rate", 24000)
                   for item in items for part in item["content"]
                   if part["type"] == "input_audio")

    def schedule_overlap(self, item, run):
        item["_listener_response"] = run.id
        fragments = (*self.preceding_overlaps(
            run, item.get("_speech_start_ms", self.clock_ms), before=item), item)
        # Reserve before scheduling: generation may finish on the next event
        # loop callback and must not release a queued tool reply in between.
        self.semantic_pending += 1
        return self.spawn(self.resolve_overlap(
            item, run, self.semantic_epoch, self.user_revision,
            self.queued_task_response, fragments,
        ))

    def rollback_unheard(self, run):
        run.abort.set()
        run.ready.set()
        if self.playback_wake is not None:
            self.playback_wake.cancel()
            self.playback_wake = None
        self.event("frankie.playback.clear", response_id=run.id)
        self.metrics["barge_ins"] += 1
        text = " ".join(c["text"] for c in run.chunks if c["end_ms"] <= run.played_ms)
        if run.item is not None:
            run.item["content"] = [{"type": "output_audio", "transcript": text}]
            run.text = text

    async def resolve_overlap(self, item, run, epoch, revision, queued_result, fragments):
        """Apply a final-utterance semantic decision, without dropping input."""
        def current():
            return (not self.closed and self.current is run
                    and epoch == self.semantic_epoch and revision == self.user_revision)

        async def replace_response():
            if not current() or run.abort.is_set():
                return
            self.rollback_unheard(run)
            while not run.done and current():
                await asyncio.sleep(0.01)
            if current() and not self.listening:
                available = set(fragments[0].get("_task_results_at_capture", ()))
                deferred = self.unhandled_task_results - available if self.queued_task_response else set()
                self.start(deferred_task_results=deferred)

        try:
            decision, observation, stats = await self.listener.classify(item, run, fragments=fragments)
            for fragment in fragments:
                for index, part in enumerate(fragment["content"]):
                    if part["type"] == "input_audio":
                        self.transcribed(fragment, index, part.get("_transcript", ""))
            action = decision.action if observation.user_text.strip() else "wait"
            apply = (self.settings["interruption_policy"] == "semantic"
                     and current() and not run.abort.is_set())
            self.event("frankie.interaction.observed", utterance_id=item["id"],
                       utterance_ids=[fragment["id"] for fragment in fragments],
                       action=action, confidence=None, evidence="final_ctc_text",
                       metrics=stats, apply=apply)
            if not apply:
                return
            if action != "wait":
                for fragment in fragments:
                    fragment["_listener_consumed"] = True
            self.event("frankie.interaction", state=action,
                       reason="Experimental semantic decision on completed speech.")
            if action == "stop":
                self.rollback_unheard(run)
                self.speech_paused = True
                self.response_revision = revision
                return
            if action in {"continue", "wait"}:
                # This utterance was acknowledged by the listener policy, not
                # left waiting for a stale client response.create to answer it.
                self.response_revision = revision
                if queued_result and self.unhandled_task_results:
                    self.queued_task_response = True
                return
            # Replanning should see the words that informed this decision as
            # well as the original neural audio features. Ordinary turns and
            # stale/observation-only decisions do not receive this supplement.
            for fragment in fragments:
                for part in fragment["content"]:
                    if part["type"] == "input_audio":
                        part["_listener_transcript"] = part.get("_transcript", "")
            await replace_response()
        except (ValueError, RuntimeError, TimeoutError) as exc:
            if not current():
                return  # Superseded/cancelled observations are routine, not UI errors.
            self.event("frankie.interaction", state="listen", reason=str(exc))
            # Failure must preserve the utterance and return control to the
            # normal response path rather than silently losing what was said.
            if self.settings["interruption_policy"] == "semantic":
                await replace_response()
        finally:
            self.semantic_pending -= 1
            self.maybe_start_task_response()

    async def receive_audio(self, encoded, playback=None):
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) % 2 or len(raw) > 2 * 24000 * 2:
            raise ValueError("Append at most two seconds of PCM16 audio.")
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        if playback is not None:
            played = base64.b64decode(playback, validate=True)
            if len(played) != len(raw):
                raise ValueError("Playback PCM must align exactly with microphone PCM.")
            system = np.frombuffer(played, dtype="<i2").astype(np.float32) / 32768
        else:
            system = np.zeros_like(samples)
        td = self.settings["turn_detection"]
        rate = self.settings["input_rate"]
        if td is None:
            if not self.manual_size:
                self.capture_context = self.next_context()
                self.capture_task_results = tuple(
                    call_id for call_id in self.task_ledger.tasks
                    if call_id in self.unhandled_task_results)
            self.manual_size += len(samples)
            if self.manual_size > 90 * rate:
                raise ValueError("Audio input exceeds 90 seconds.")
            self.manual.append(samples)
            return
        self.received_ms += len(samples) * 1000 / rate
        hop = rate * 32 // 1000
        self.tail = np.concatenate([self.tail, samples])
        self.system_tail = np.concatenate([self.system_tail, system])
        while len(self.tail) >= hop:
            frame, self.tail = self.tail[:hop], self.tail[hop:]
            played, self.system_tail = self.system_tail[:hop], self.system_tail[hop:]
            self.clock_ms = round(self.received_ms - len(self.tail) * 1000 / rate)
            self.metrics["input_frames"] += 1
            if self.current and self.current.visible and not self.current.done:
                self.metrics["input_frames_during_output"] += 1
            user_frame = resample(frame, rate, 16000)
            probability = self.vad(user_frame)
            if self.turn is not None:
                self.turn.append(user_frame, resample(played, rate, 16000))
            voiced = probability >= td.get("threshold", 0.5)
            self.voice_run = self.voice_run + 1 if voiced else 0
            if voiced and not self.listening:
                self.invalidate_semantics()
                self.capture_task_results = tuple(
                    call_id for call_id in self.task_ledger.tasks
                    if call_id in self.unhandled_task_results)
                self.overlap_run = self.audible_response()
                self.overlap_prefix_ms = self.fragment_audio_ms(self.preceding_overlaps(
                    self.overlap_run, self.clock_ms)) if self.overlap_run is not None else 0
                self.listening = True
                self.capture_context = self.next_context()
                self.speech_start_ms = self.clock_ms
                self.frames = list(self.pre)
                self.silence = 0
                self.speech_id = identifier("item")
                self.event(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=max(0, self.clock_ms - 32 * (len(self.pre) + 1)),
                    item_id=self.speech_id,
                )
            if self.listening:
                self.frames.append(frame)
                buffered_samples = sum(len(x) for x in self.frames)
                if (self.overlap_run is not None
                        and self.settings["interruption_policy"] == "semantic"
                        and self.overlap_prefix_ms + buffered_samples * 1000 / rate
                        > MAX_OBSERVATION_SECONDS * 1000):
                    # Match the listener's six-second acoustic budget, including
                    # endpoint/pre-roll padding and unresolved earlier fragments.
                    # Yield now instead of talking
                    # over a long user turn until its eventual endpoint. Keep
                    # all PCM for the ordinary response path after this fallback.
                    self.invalidate_semantics()
                    if (not self.overlap_run.abort.is_set()
                            and not self.overlap_run.playback_finished):
                        self.rollback_unheard(self.overlap_run)
                    self.overlap_run = None
                    self.overlap_prefix_ms = 0
                    self.event("frankie.interaction", state="yield", fallback="audio_budget",
                               reason="Long user turn; returning to ordinary turn handling.")
                if voiced:
                    self.silence = 0
                    self.last_speech_ms = self.clock_ms
                    if self.spec:
                        self.discard_spec()
                    active = self.current
                    if self.voice_run >= 3 and active and active.visible:
                        if (
                            not active.abort.is_set()
                            and (not active.done or self.audible_response() is active)
                            and td.get("interrupt_response", True)
                            and not (self.settings["interruption_policy"] == "semantic"
                                     and self.overlap_run is not None)
                        ):
                            if active.emitted_ms > 0:
                                self.rollback_unheard(active)
                            else:
                                self.cancel()
                                self.metrics["barge_ins"] += 1
                                self.event("frankie.playback.clear", response_id=active.id)
                        # The client may already have cancelled on speech_started.
                        if active.abort.is_set():
                            self.merge_resumed(active)
                else:
                    self.silence += 32
                    can_start = (
                        self.current is None
                        or self.current.done
                        or self.current.abort.is_set()
                    )
                    semantic_overlap = self.overlap_run is not None and self.settings["interruption_policy"] == "semantic"
                    if semantic_overlap:
                        can_start = False
                    if (
                        self.silence >= 96
                        and self.spec is None
                        and can_start
                        and td.get("create_response", True)
                    ):
                        self.start(
                            self.audio_item(
                                np.concatenate(self.frames),
                                item_id=self.speech_id,
                                speech_end_ms=self.last_speech_ms,
                            ),
                            tentative=True,
                        )
                    if self.silence >= max(
                        160, td.get("silence_duration_ms", 320)
                    ) and (
                        self.turn is None
                        or self.turn.release(self.clock_ms, self.silence)
                    ):
                        self.event(
                            "input_audio_buffer.speech_stopped",
                            audio_end_ms=self.clock_ms,
                            frankie_last_speech_ms=self.last_speech_ms,
                            item_id=self.speech_id,
                        )
                        if self.spec:
                            observed_item = self.spec.input
                            self.show(self.spec)
                            self.spec = None
                            # Observation mode retains normal VAD/speculation.
                            # Its input still needs a listener observation when
                            # the speculative response is committed here.
                            if (observed_item is not None and self.overlap_run is not None
                                    and self.settings["interruption_policy"] == "observe"):
                                self.schedule_overlap(observed_item, self.overlap_run)
                        else:
                            item = self.audio_item(
                                np.concatenate(self.frames),
                                item_id=self.speech_id,
                                speech_end_ms=self.last_speech_ms,
                            )
                            self.items.append(item)
                            self.accept_input(semantic_overlap=semantic_overlap)
                            self.event(
                                "input_audio_buffer.committed",
                                item_id=item["id"],
                                previous_item_id=None,
                            )
                            self.event(
                                "conversation.item.created",
                                item=public(item),
                                previous_item_id=None,
                            )
                            if self.overlap_run is not None and self.settings["interruption_policy"] in {"observe", "semantic"}:
                                self.schedule_overlap(item, self.overlap_run)
                            if td.get("create_response", True) and can_start:
                                self.start()
                            elif not semantic_overlap:
                                self.spawn(self.transcribe_item(item))
                        self.frames = []
                        self.listening = False
                        self.capture_context = (0, "")
                        self.silence = 0
                        self.overlap_run = None
                        self.overlap_prefix_ms = 0
                if sum(len(x) for x in self.frames) > 90 * rate:
                    self.discard_spec()
                    self.frames = []
                    self.listening = False
                    self.capture_context = (0, "")
                    raise ValueError("Utterance exceeds 90 seconds.")
            self.pre.append(frame)

    async def handle(self, event):
        kind = event["type"]
        if kind == "frankie.input_context.update":
            revision, text = event["revision"], event["text"]
            if (
                type(revision) is not int
                or not self.latest_context[0] < revision < 2**64
                or not isinstance(text, str)
                or len(text.encode()) > 16384
            ):
                raise ValueError("Invalid input context.")
            self.latest_context = (
                revision,
                text.replace("{{frankie_media_", "{{frankie-media_"),
            )
            self.event("frankie.input_context.updated", revision=revision)
        elif kind == "session.update":
            if self.current and not self.current.done:
                raise ValueError(
                    "Wait for the current response before changing session settings."
                )
            settings = event["session"]
            new = copy.deepcopy(self.settings)
            extensions = settings.get("frankie", {})
            if "playback_feedback" in extensions:
                if type(extensions["playback_feedback"]) is not bool:
                    raise ValueError("playback_feedback must be a boolean.")
                new["playback_feedback"] = extensions["playback_feedback"]
            if "assistant_name" in extensions:
                name = extensions["assistant_name"]
                if not isinstance(name, str) or not 1 <= len(name.strip()) <= 64:
                    raise ValueError("assistant_name must contain 1 to 64 characters.")
                new["assistant_name"] = name.strip()
            if "interruption_policy" in extensions:
                policy = extensions["interruption_policy"]
                if policy not in {"vad", "observe", "semantic"}:
                    raise ValueError("Use vad, observe, or semantic interruption policy.")
                if policy != "vad" and self.listener is None:
                    raise ValueError("Semantic listener is unavailable in this server.")
                new["interruption_policy"] = policy
            if "background_tasks" in extensions:
                if type(extensions["background_tasks"]) is not bool:
                    raise ValueError("background_tasks must be a boolean.")
                new["background_tasks"] = extensions["background_tasks"]
                if not new["background_tasks"] and any(
                    task.status == "running" for task in self.task_ledger.tasks.values()
                ):
                    raise ValueError("Finish or cancel background tasks before disabling them.")
            for key in (
                "instructions",
                "output_modalities",
                "temperature",
                "tools",
                "context",
            ):
                if key in settings:
                    new[key] = settings[key]
            if "max_output_tokens" in settings:
                new["max_output_tokens"] = int(settings["max_output_tokens"])
            if not 1 <= new["max_output_tokens"] <= 8192:
                raise ValueError("max_output_tokens must be between 1 and 8192.")
            if (
                not set(new["output_modalities"]) <= {"audio", "text"}
                or not new["output_modalities"]
            ):
                raise ValueError("Choose text and/or audio output.")
            thinking = settings.get(
                "thinking", settings.get("reasoning", {}).get("effort", new["thinking"])
            )
            if "enable_thinking" in settings:
                thinking = "medium" if settings["enable_thinking"] else "off"
            if thinking == "none":
                thinking = "off"
            from .sampling import THINKING_BUDGETS, brain_sampler
            if thinking not in THINKING_BUDGETS:
                raise ValueError("Unknown thinking level.")
            new["thinking"] = thinking
            brain_sampler(new, realtime=True)
            audio = settings.get("audio", {}).get("input", {})
            if "turn_detection" in audio:
                new["turn_detection"] = audio["turn_detection"]
            if "turn_detection" in settings:
                new["turn_detection"] = settings["turn_detection"]
            if "format" in audio:
                rate = int(audio["format"].get("rate", 24000))
                if rate not in {16000, 24000}:
                    raise ValueError("PCM input rate must be 16000 or 24000.")
                new["input_rate"] = rate
            if new == self.settings:
                self.event("session.updated", session=self.info())
                return
            if self.listening or self.manual_size:
                raise ValueError("Clear pending audio before changing settings.")
            if (new["interruption_policy"] != "vad"
                    and self.settings["interruption_policy"] == "vad"):
                await self.loop.run_in_executor(
                    self.executor, self.listener.service.warm_listener
                )
            changed_prefix = any(
                new[k] != self.settings[k]
                for k in ("instructions", "thinking", "tools")
            )
            if changed_prefix:
                await self.loop.run_in_executor(
                    self.executor, lambda: self.engine.warm(new, session_id=self.id)
                )
            if new["background_tasks"] and not self.settings["background_tasks"]:
                ledger = TaskLedger()
                for item in self.items:
                    if item["type"] == "function_call":
                        ledger.register(item, self.input_revision)
                    elif item["type"] == "function_call_output":
                        ledger.complete(item["call_id"])
                self.task_ledger = ledger
            self.settings = new
            self.reset_detectors()
            self.event("session.updated", session=self.info())
        elif kind == "conversation.item.create":
            item = copy.deepcopy(event["item"])
            interrupt = False
            item.setdefault("id", identifier("item"))
            if any(i["id"] == item["id"] for i in self.items):
                raise ValueError("Duplicate conversation item id.")
            if self.current and not self.current.done:
                if self.settings["background_tasks"] and item["type"] == "function_call_output":
                    pass  # This result is visible only to a subsequent response snapshot.
                elif self.settings["background_tasks"] and item.get("role") == "user":
                    interrupt = True
                else:
                    raise ValueError(
                        "Cancel the active response before inserting a conversation item."
                    )
            if item["type"] == "message":
                if item["role"] not in {"user", "assistant", "system"}:
                    raise ValueError("Invalid message role.")
                for part in item.get("content", []):
                    if part["type"] == "input_image":
                        from mtplx.vision.media import (
                            image_bytes_from_url,
                            validate_image_detail,
                        )
                        if item["role"] != "user":
                            raise ValueError("Images belong in user messages.")
                        validate_image_detail(part.get("detail", "auto"))
                        part["_bytes"] = image_bytes_from_url(
                            part["image_url"], max_bytes=12 * 1024**2, inline_only=True)
                    elif part["type"] == "input_audio":
                        data = base64.b64decode(part.pop("audio"), validate=True)
                        if len(data) % 2 or len(data) > 90 * 24000 * 2:
                            raise ValueError("Invalid or oversized audio input.")
                        part["_pcm"] = (
                            np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
                        )
                        part["_rate"] = self.settings["input_rate"]
                    elif part["type"] not in {
                        "input_text",
                        "text",
                        "audio",
                        "output_text",
                        "output_audio",
                    }:
                        raise ValueError("Unsupported content type.")
            elif item["type"] == "function_call":
                if not isinstance(json.loads(item["arguments"]), dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                if not item.get("name") or not item.get("call_id"):
                    raise ValueError("Tool history requires a name and call_id.")
                if any(
                    i.get("call_id") == item["call_id"] and i["type"] == "function_call"
                    for i in self.items
                ):
                    raise ValueError("Duplicate tool call id.")
                if self.settings["background_tasks"]:
                    task = self.task_ledger.register(item, self.input_revision)
                    self.event("frankie.task.updated", task=task.public())
            elif item["type"] == "function_call_output":
                if not any(
                    i.get("call_id") == item["call_id"] and i["type"] == "function_call"
                    for i in self.items
                ):
                    raise ValueError("Unknown tool call.")
                if any(
                    i.get("call_id") == item["call_id"]
                    and i["type"] == "function_call_output"
                    for i in self.items
                ):
                    raise ValueError("Tool result already supplied.")
                if self.settings["background_tasks"]:
                    if not isinstance(item.get("output"), str):
                        raise ValueError("Tool output must be a string.")
                    task, accepted = self.task_ledger.complete(item["call_id"])
                    if accepted:
                        self.input_revision += 1
                        self.unhandled_task_results.add(item["call_id"])
                    else:
                        item["_task_discarded"] = True
                    self.event(
                        "frankie.task.updated", task=task.public(), result_discarded=not accepted
                    )
            else:
                raise ValueError("Unsupported conversation item type.")
            if item["type"] == "message" and item["role"] == "user":
                if interrupt:
                    if self.current.emitted_ms > 0:
                        self.rollback_unheard(self.current)
                    else:
                        self.cancel()
                        self.event("frankie.playback.clear", response_id=self.current.id)
                self.attach_context(item, self.next_context())
                self.accept_input()
            self.items.append(item)
            self.event(
                "conversation.item.created", item=public(item), previous_item_id=None
            )
            if any(p["type"] == "input_audio" for p in item.get("content", [])):
                self.spawn(self.transcribe_item(item))
        elif kind == "response.create":
            if self.speech_paused:
                self.event("frankie.response.skipped", reason="user_requested_silence")
                return
            if self.settings["background_tasks"]:
                # An explicit response request reserves available results even
                # when an already captured user turn must be answered first.
                if self.unhandled_task_results:
                    self.queued_task_response = True
                if self.unhandled_task_results and (
                    self.listening or self.manual_size or self.semantic_pending
                ):
                    self.event("frankie.response.queued", reason="user_turn_pending")
                    return
                if self.semantic_pending:
                    self.event("frankie.response.skipped", reason="semantic_turn_pending")
                    return
                if (self.unhandled_task_results and self.audible_response() is not None
                        and (not self.user_revision or self.response_revision == self.user_revision)):
                    self.event("frankie.response.queued", reason="playback_pending")
                    self.maybe_start_task_response()
                    return
                if self.current and not self.current.done and not self.current.abort.is_set():
                    if self.unhandled_task_results:
                        self.event("frankie.response.queued", reason="background_task_result")
                        return
                    if self.response_revision == self.user_revision:
                        self.event("frankie.response.skipped", reason="no_new_input")
                        return
                elif self.response_revision == self.user_revision and not self.unhandled_task_results:
                    self.event("frankie.response.skipped", reason="no_new_input")
                    return
            self.start()
        elif kind == "response.cancel":
            run = self.current
            if event.get("response_id") and (not run or event["response_id"] != run.id):
                raise ValueError("Response id is not active.")
            self.queued_task_response = False
            self.invalidate_semantics()
            self.speech_paused = True
            audible = self.audible_response()
            if audible is not None:
                self.rollback_unheard(audible)
            else:
                self.cancel()
        elif kind == "input_audio_buffer.append":
            await self.receive_audio(event["audio"], event.get("playback"))
        elif kind == "input_audio_buffer.commit":
            if not self.manual:
                raise ValueError("Input audio buffer is empty.")
            item = self.audio_item(np.concatenate(self.manual))
            self.manual = []
            self.manual_size = 0
            self.capture_context = (0, "")
            self.items.append(item)
            self.accept_input()
            self.event(
                "input_audio_buffer.committed",
                item_id=item["id"],
                previous_item_id=None,
            )
            self.event(
                "conversation.item.created", item=public(item), previous_item_id=None
            )
            self.spawn(self.transcribe_item(item))
        elif kind == "input_audio_buffer.clear":
            self.discard_spec()
            self.manual = []
            self.manual_size = 0
            self.frames = []
            self.listening = False
            self.capture_context = (0, "")
            self.reset_detectors()
            self.event("input_audio_buffer.cleared")
            self.maybe_start_task_response()
        elif kind == "frankie.task.cancel":
            if not self.settings["background_tasks"]:
                raise ValueError("Background tasks are not enabled.")
            task, changed = self.task_ledger.cancel(event["call_id"], event.get("reason", "cancelled"))
            if changed:
                for item in self.items:
                    if not task.consumed and item["type"] == "function_call_output" and item["call_id"] == task.call_id:
                        item["_task_discarded"] = True
                notice = task_notice(task.call_id, task.name, status=task.status)
                notice["id"] = identifier("item")
                # This is an explicit harness status update, not a completion
                # claim. It preserves the point at which cancellation occurred.
                notice["role"] = "system"
                self.items.append(notice)
                self.event("conversation.item.created", item=notice, previous_item_id=None)
                self.unhandled_task_results.discard(task.call_id)
                if not self.unhandled_task_results:
                    self.queued_task_response = False
            self.event(
                "frankie.task.updated", task=task.public(),
                cancellation_requested=changed and not task.received,
            )
        elif kind in {"frankie.playback.position", "frankie.playback.finished"}:
            run = self.playback_runs.get(event["item_id"])
            ms = event["audio_end_ms"]
            if (
                run is None or event.get("response_id") != run.id
                or type(ms) is not int or not run.played_ms <= ms <= run.emitted_ms + 1
            ):
                raise ValueError("Invalid or stale playback position.")
            if kind == "frankie.playback.finished":
                if not run.done or ms < run.emitted_ms - 1:
                    raise ValueError("Playback cannot finish before response completion and drain.")
                run.playback_finished = True
            run.played_ms = ms
            self.settings["playback_feedback"] = True
            if kind == "frankie.playback.finished":
                self.maybe_start_task_response()
        elif kind == "conversation.item.truncate":
            item = next((i for i in self.items if i["id"] == event["item_id"]), None)
            run = self.playback_runs.get(event["item_id"], self.current)
            if item is None and run and run.merged and run.item_id == event["item_id"]:
                item = run.item
            if item is None or item.get("role") != "assistant":
                raise ValueError("Unknown assistant audio item.")
            if run and run.item_id == item["id"]:
                ms = max(0, int(event["audio_end_ms"]))
                if not (run.playback_finished and ms >= run.emitted_ms - 1):
                    run.abort.set()
                    run.ready.set()
                    run.played_ms = ms
                    text = " ".join(c["text"] for c in run.chunks if c["end_ms"] <= ms)
                    item["content"] = [
                        {
                            "type": "output_audio",
                            "transcript": text + " [interrupted by the user]",
                        }
                    ]
                    run.text = text
            self.event(
                "conversation.item.truncated",
                item_id=item["id"],
                content_index=0,
                audio_end_ms=event["audio_end_ms"],
            )
        elif kind == "conversation.item.retrieve":
            item = next((i for i in self.items if i["id"] == event["item_id"]), None)
            if item is None:
                raise ValueError("Unknown item.")
            self.event("conversation.item.retrieved", item=public(item))
        elif kind == "frankie.voice.update":
            if self.current and not self.current.done:
                raise ValueError("Wait for speech to finish before changing the voice.")
            data = base64.b64decode(event["wav"], validate=True)
            if len(data) > 12 * 1024**2:
                raise ValueError("Voice reference exceeds 12 MiB.")
            await self.loop.run_in_executor(
                self.executor,
                lambda: self.engine.audio.voice_from_wav(data, event.get("transcript")),
            )
            self.event("frankie.voice.updated")
        else:
            raise ValueError("Unsupported event type: " + kind)

    async def close(self):
        self.closed = True
        if self.playback_wake is not None:
            self.playback_wake.cancel()
            self.playback_wake = None
        if self.listener is not None:
            self.listener.cancel()
        self.cancel()
        self.discard_spec()
        if self.turn is not None:
            await asyncio.to_thread(self.turn.close)
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.loop.run_in_executor(
            self.executor, lambda: self.engine.release(self.id)
        )
