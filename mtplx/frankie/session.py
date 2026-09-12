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


class Session:
    def __init__(self, engine, executor, websocket):
        self.engine, self.executor, self.ws = engine, executor, websocket
        self.loop = asyncio.get_running_loop()
        self.id = identifier("sess")
        self.items = []
        self.current = None
        self.spec = None
        self.closed = False
        self.tasks = set()
        self.outgoing = asyncio.Queue(maxsize=2048)
        self.settings = {
            "instructions": "You are Frankie. Be helpful and concise. Use natural spoken sentences.",
            "output_modalities": ["audio"],
            "max_output_tokens": 512,
            "temperature": 0.7,
            "thinking": "off",
            "context": 131072,
            "tools": [],
            "input_rate": 24000,
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
        run.committed_at = time.monotonic()
        audio = "audio" in run.settings["output_modalities"]
        run.item = {
            "id": run.item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [
                {"type": "audio", "transcript": ""}
                if audio
                else {"type": "text", "text": ""}
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
        common = {
            "response_id": run.id,
            "item_id": run.item_id,
            "output_index": 0,
            "content_index": 0,
        }
        if kind == "audio":
            self.event("response.output_audio.delta", delta=value, **common)
        elif kind == "text":
            run.text += value
            part = run.item["content"][0]
            part["transcript" if part["type"] == "audio" else "text"] = run.text
            self.event(
                "response.output_audio_transcript.delta"
                if part["type"] == "audio"
                else "response.output_text.delta",
                delta=value,
                **common,
            )
        elif kind == "chunk":
            run.chunks.append(value)

    def start(self, input_item=None, *, tentative=False):
        if (
            self.current is not None
            and not self.current.done
            and not self.current.abort.is_set()
        ):
            raise ValueError("A response is already active.")
        run = Response(input=input_item, settings=copy.deepcopy(self.settings))
        self.current = run
        history = list(self.items) + (
            [input_item]
            if input_item is not None and input_item not in self.items
            else []
        )
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
        except Exception as exc:
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
            from mtplx.server.omlx_bridge.tool_calling import parse_tool_calls

            parsed = parse_tool_calls(
                result["raw_text"].rsplit("</think>", 1)[-1],
                self.engine.tokenizer,
                run.settings["tools"],
            )
            for call in parsed.tool_calls or []:
                function = call["function"]
                item = {
                    "id": identifier("item"),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call.get("id", identifier("call")),
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
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

    def cancel(self):
        run = self.current
        if run is not None and not run.done:
            run.abort.set()
            run.ready.set()
        return run

    def discard_spec(self):
        if self.spec is None:
            return
        self.spec.abort.set()
        self.spec.ready.set()
        self.metrics["speculation_aborts"] += 1
        self.event("frankie.speculation.aborted", response_id=self.spec.id)
        self.spec = None

    def audio_item(self, pcm, *, item_id=None):
        return {
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

    def reset_detectors(self):
        self.vad.reset()
        self.tail = np.empty(0, dtype=np.float32)
        self.system_tail = np.empty(0, dtype=np.float32)
        self.pre.clear()
        if self.turn is not None:
            self.turn.reset(self.received_ms)

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
            self.clock_ms = int(round(self.received_ms - len(self.tail) * 1000 / rate))
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
                self.listening = True
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
                if voiced:
                    self.silence = 0
                    self.last_speech_ms = self.clock_ms
                    if self.spec:
                        self.discard_spec()
                    active = self.current
                    if (
                        self.voice_run >= 3
                        and active
                        and active.visible
                        and not active.done
                        and not active.abort.is_set()
                        and td.get("interrupt_response", True)
                    ):
                        self.cancel()
                        self.metrics["barge_ins"] += 1
                        self.event("frankie.playback.clear", response_id=active.id)
                        # A quick resumed utterance belongs to the same user
                        # input; remove the speculative reply from history.
                        if (
                            active.input
                            and time.monotonic() - active.committed_at < 0.7
                        ):
                            self.frames.insert(0, active.input["content"][0]["_pcm"])
                            self.items = [
                                i
                                for i in self.items
                                if i["id"] not in {active.input["id"], active.item_id}
                            ]
                            self.event("frankie.input.merged", response_id=active.id)
                else:
                    self.silence += 32
                    can_start = (
                        self.current is None
                        or self.current.done
                        or self.current.abort.is_set()
                    )
                    if (
                        self.silence >= 96
                        and self.spec is None
                        and can_start
                        and td.get("create_response", True)
                    ):
                        self.start(
                            self.audio_item(
                                np.concatenate(self.frames), item_id=self.speech_id
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
                            self.show(self.spec)
                            self.spec = None
                        else:
                            item = self.audio_item(
                                np.concatenate(self.frames), item_id=self.speech_id
                            )
                            self.items.append(item)
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
                            if td.get("create_response", True) and can_start:
                                self.start()
                        self.frames = []
                        self.listening = False
                        self.silence = 0
                if sum(len(x) for x in self.frames) > 90 * rate:
                    self.discard_spec()
                    self.frames = []
                    self.listening = False
                    raise ValueError("Utterance exceeds 90 seconds.")
            self.pre.append(frame)

    async def handle(self, event):
        kind = event["type"]
        if kind == "session.update":
            if self.current and not self.current.done:
                raise ValueError(
                    "Wait for the current response before changing session settings."
                )
            settings = event["session"]
            new = copy.deepcopy(self.settings)
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
            if thinking not in {"off", "minimal", "low", "medium", "high"}:
                raise ValueError("Unknown thinking level.")
            new["thinking"] = thinking
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
            changed_prefix = any(
                new[k] != self.settings[k]
                for k in ("instructions", "thinking", "tools")
            )
            if changed_prefix:
                await self.loop.run_in_executor(
                    self.executor, lambda: self.engine.warm(new, session_id=self.id)
                )
            self.settings = new
            self.reset_detectors()
            self.event("session.updated", session=self.info())
        elif kind == "conversation.item.create":
            if self.current and not self.current.done:
                raise ValueError(
                    "Cancel the active response before inserting a conversation item."
                )
            item = copy.deepcopy(event["item"])
            item.setdefault("id", identifier("item"))
            if any(i["id"] == item["id"] for i in self.items):
                raise ValueError("Duplicate conversation item id.")
            if item["type"] == "message":
                if item["role"] not in {"user", "assistant", "system"}:
                    raise ValueError("Invalid message role.")
                for part in item.get("content", []):
                    if part["type"] == "input_image":
                        url = part["image_url"]
                        url = url["url"] if isinstance(url, dict) else url
                        if not url.startswith("data:image/"):
                            raise ValueError("Supply image bytes as a data URL.")
                        part["_bytes"] = base64.b64decode(
                            url.split(",", 1)[1], validate=True
                        )
                        if len(part["_bytes"]) > 12 * 1024**2:
                            raise ValueError("Image exceeds 12 MiB.")
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
            else:
                raise ValueError("Unsupported conversation item type.")
            self.items.append(item)
            self.event(
                "conversation.item.created", item=public(item), previous_item_id=None
            )
        elif kind == "response.create":
            self.start()
        elif kind == "response.cancel":
            run = self.current
            if event.get("response_id") and (not run or event["response_id"] != run.id):
                raise ValueError("Response id is not active.")
            self.cancel()
        elif kind == "input_audio_buffer.append":
            await self.receive_audio(event["audio"], event.get("playback"))
        elif kind == "input_audio_buffer.commit":
            if not self.manual:
                raise ValueError("Input audio buffer is empty.")
            item = self.audio_item(np.concatenate(self.manual))
            self.manual = []
            self.manual_size = 0
            self.items.append(item)
            self.event(
                "input_audio_buffer.committed",
                item_id=item["id"],
                previous_item_id=None,
            )
            self.event(
                "conversation.item.created", item=public(item), previous_item_id=None
            )
        elif kind == "input_audio_buffer.clear":
            self.discard_spec()
            self.manual = []
            self.manual_size = 0
            self.frames = []
            self.listening = False
            self.reset_detectors()
            self.event("input_audio_buffer.cleared")
        elif kind == "conversation.item.truncate":
            item = next((i for i in self.items if i["id"] == event["item_id"]), None)
            if item is None or item.get("role") != "assistant":
                raise ValueError("Unknown assistant audio item.")
            run = self.current
            if run and run.item_id == item["id"]:
                self.cancel()
                ms = max(0, int(event["audio_end_ms"]))
                text = " ".join(c["text"] for c in run.chunks if c["end_ms"] <= ms)
                item["content"] = [
                    {"type": "audio", "transcript": text + " [interrupted by the user]"}
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
        self.cancel()
        self.discard_spec()
        if self.turn is not None:
            await asyncio.to_thread(self.turn.close)
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.loop.run_in_executor(
            self.executor, lambda: self.engine.release(self.id)
        )
