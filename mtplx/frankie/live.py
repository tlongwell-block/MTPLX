"""GPT-Live sessions for Frankie: the WebRTC dialect that goose's live voice speaks.

    POST /v1/live/sessions             {session, transport: {type: webrtc, sdp}}
                                       -> 201 {session: {id}, transport: {type: webrtc, sdp}}
    WS   /v1/live/sessions/{id}/attach  sideband: transcripts, delegation, commentary

Each live session drives one ordinary realtime ``Session`` in this process, so
turn-taking, barge-in, listening sounds and 48 kHz speech are Frankie's own. This
module only moves audio between WebRTC and that session and maps its events.

Delegation is one internal background tool, ``delegate``. Its call becomes
``session.delegation.created``; the client's ``session.commentary.append`` for that
delegation becomes the tool's output, which Frankie then says in its own words.
Commentary is context for the voice, never text it reads out.
"""

import asyncio
import base64
import fractions
import hmac
import json
import time
from collections import deque

import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

from .session import Session, identifier

MAX_BODY_BYTES = 1 << 20
MAX_SDP_BYTES = 64 * 1024
MAX_INPUT_MESSAGES = 128
MAX_COMMENTARY_BYTES = 64 * 1024
MAX_CONTEXT_BYTES = 16384  # Frankie's input context limit.
MAX_PENDING_EVENTS = 256
SETUP_TIMEOUT_S = 9  # goose gives signaling 10 s.
ATTACH_TIMEOUT_S = 15
EAR_RATE = 24000
FRAME_MS = 20
PREROLL_MS = 120  # Frankie's page worklet holds this much before starting a reply.

DELEGATE = {
    "type": "function",
    "name": "delegate",
    "description": ("Hand the user's request to Goose, which has tools and can do real work. "
                    "Goose hears this conversation, so this takes no arguments. "
                    "Its result arrives later as this tool's output."),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}
STYLE = ("Everything you say is spoken aloud. Use plain, unformatted conversational "
         "language, and keep replies brief.")
DELEGATE_GUIDE = ("To delegate to Goose, call the delegate tool. Before the call, briefly say you're on it. "
                  "When its output arrives, give the user the result in your own words, in one to three "
                  "spoken sentences. Never read out formatting, code, or long identifiers.")
FRANKIE = {"background_tasks": True, "playback_feedback": True, "playback_pause": False,
           "backchannels": True}
LISTENER = {"interruption_policy": "semantic", "streaming_listener": "backchannel"}
CONTEXT_CHANNELS = {f"session.{c}.append": c for c in ("instructions", "thinking", "commentary")}
CONTEXT_LABELS = {"instructions": "Instructions from Goose", "thinking": "Goose is thinking",
                  "commentary": "Note from Goose"}
ROLES = {"user": ("user", "input_text"), "assistant": ("assistant", "output_text"),
         "developer": ("system", "input_text")}


def parse(data):
    """Validate a session creation body; return (session, offer SDP)."""
    if not isinstance(data, dict):
        raise ValueError("Send a JSON object.")
    session, transport = data.get("session"), data.get("transport")
    if not isinstance(session, dict) or not isinstance(transport, dict):
        raise ValueError("Send session and transport objects.")
    sdp = transport.get("sdp")
    if (transport.get("type") != "webrtc" or not isinstance(sdp, str) or not sdp.strip()
            or len(sdp.encode()) > MAX_SDP_BYTES):
        raise ValueError("transport must be webrtc with an SDP offer under 64 KiB.")
    model = session.get("model")
    if not isinstance(model, str) or not 0 < len(model.strip()) <= 128:
        raise ValueError("session.model must be a model name.")
    instructions = session.get("instructions", "")
    if not isinstance(instructions, str):
        raise ValueError("session.instructions must be a string.")
    if session.get("delegation", {"type": "client"}) != {"type": "client"}:
        raise ValueError("Only client delegation is supported.")
    messages = session.get("input", [])
    if not isinstance(messages, list) or len(messages) > MAX_INPUT_MESSAGES:
        raise ValueError(f"session.input must be a list of at most {MAX_INPUT_MESSAGES} messages.")
    items = []
    for message in messages:
        message = message if isinstance(message, dict) else {}
        role, kind = ROLES.get(message.get("role"), (None, None))
        content = message.get("content")
        if (role is None or message.get("type") != "message" or not isinstance(content, list)
                or not all(isinstance(p, dict) and p.get("type") in {"input_text", "output_text"}
                           and isinstance(p.get("text"), str) for p in content)):
            raise ValueError("session.input holds user, assistant or developer text messages.")
        items.append({"type": "message", "role": role,
                      "content": [{"type": kind, "text": p["text"]} for p in content]})
    return {"model": model.strip(), "instructions": instructions, "items": items}, sdp


def take(chunks, n):
    """Pop up to n samples from a deque of int16 arrays."""
    parts, got = [], 0
    while chunks and got < n:
        head = chunks[0]
        part = head[:n - got]
        parts.append(part)
        got += len(part)
        if len(part) == len(head):
            chunks.popleft()
        else:
            chunks[0] = head[len(part):]
    return np.concatenate(parts) if parts else np.empty(0, np.int16)


def pcm(encoded):
    return np.frombuffer(base64.b64decode(encoded), dtype="<i2").copy()


class Speaker(MediaStreamTrack):
    """Frankie's voice on the media clock, keeping the queue Frankie's page worklet keeps.

    It reports what was actually sent as playback feedback, as the page does from its
    device clock, and keeps a 24 kHz copy as the ear's echo reference.
    """

    kind = "audio"

    def __init__(self, rate, feedback):
        super().__init__()
        if rate not in (24000, 48000):
            raise ValueError("Frankie speaks at 24 or 48 kHz.")
        self.rate, self.feedback = rate, feedback
        self.frame = rate * FRAME_MS // 1000
        self.queue, self.queued = deque(), 0
        self.item, self.played, self.started, self.ended = None, 0, False, False
        self.paused, self.blocked = None, set()
        self.aside, self.aside_id = deque(), None
        self.heard, self.heard_size = deque(), 0
        self.start, self.timestamp, self.tick = None, 0, 0

    def ms(self):
        return self.played * 1000 // self.rate

    def audio(self, response_id, item_id, samples):
        if item_id in self.blocked:
            return
        if self.aside_id:
            self.blocked.add(self.aside_id)
        self.aside.clear()
        if self.item and self.item[1] != item_id:
            self.clear(None)  # A new reply replaces one still queued.
        if self.item is None:
            self.item, self.played, self.started, self.ended = (response_id, item_id), 0, False, False
        if self.queued + len(samples) <= 30 * self.rate:
            self.queue.append(samples)
            self.queued += len(samples)

    def done(self, item_id):
        if self.item and self.item[1] == item_id:
            self.ended = True
            self.finish()

    def clear(self, response_id):
        if self.aside_id:
            self.blocked.add(self.aside_id)
        self.aside.clear()
        if self.item is None or (response_id and self.item[0] != response_id):
            return
        self.blocked.add(self.item[1])
        self.queue.clear()
        self.queued = 0
        self.feedback("truncate", self.item, self.ms())
        self.item = None

    def pause(self, response_id, paused):
        if self.item and self.item[0] == response_id:
            self.paused = response_id if paused else None
            if paused:
                self.feedback("position", self.item, self.ms())

    def backchannel(self, item_id, samples):
        if self.queued or item_id in self.blocked or (self.aside and self.aside_id != item_id):
            return
        self.aside_id = item_id
        if sum(map(len, self.aside)) + len(samples) <= self.rate * 128 // 100:
            self.aside.append(samples)

    def finish(self):
        # Never truncate a fully heard reply; an underrun stays interruptible.
        if self.item and self.ended and not self.queued:
            self.feedback("finished", self.item, self.ms())
            self.item = None

    def echo(self, out):
        ref = out if self.rate == EAR_RATE else (out.reshape(-1, 2).astype(np.int32).sum(1) // 2).astype(np.int16)
        self.heard.append(ref)
        self.heard_size += len(ref)
        while self.heard_size - len(self.heard[0]) >= EAR_RATE:
            self.heard_size -= len(self.heard.popleft())

    def heard_for(self, n):
        """The speech sent while the next n microphone samples were captured (zeros if none)."""
        got = take(self.heard, n)
        self.heard_size -= len(got)
        return np.concatenate([got, np.zeros(n - len(got), np.int16)])

    def next_frame(self):
        out = np.zeros(self.frame, np.int16)
        if self.item and self.item[0] != self.paused and self.queued:
            self.started = self.started or self.ended or self.queued >= self.rate * PREROLL_MS // 1000
            if self.started:
                got = take(self.queue, self.frame)
                out[:len(got)] = got
                self.queued -= len(got)
                self.played += len(got)
        elif not self.queued and self.aside:
            got = take(self.aside, self.frame)
            out[:len(got)] = got
        self.echo(out)
        self.tick += self.frame
        if self.tick >= self.rate // 4:
            self.tick = 0
            if self.item and self.started:
                self.feedback("position", self.item, self.ms())
        self.finish()
        return out

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        if self.start is None:
            self.start = time.time()
        else:
            self.timestamp += self.frame
            wait = self.start + self.timestamp / self.rate - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        frame = AudioFrame(format="s16", layout="mono", samples=self.frame)
        frame.planes[0].update(self.next_frame().tobytes())
        frame.sample_rate, frame.pts = self.rate, self.timestamp
        frame.time_base = fractions.Fraction(1, self.rate)
        return frame


class Link:
    """The websocket a realtime Session writes to: its events come here instead."""

    def __init__(self, live):
        self.live = live

    async def send_json(self, event):
        await self.live.frankie_event(event)

    async def close(self, code=1000, reason=""):
        self.live.spawn(self.live.close("server_error"))


class LiveSession:
    def __init__(self, engine, executor, listener_service=None, on_end=lambda live: None):
        self.id = identifier("live")
        self.session = Session(engine, executor, Link(self))
        if listener_service is not None:
            from .perception import RealtimeListener
            self.session.listener = RealtimeListener(self.session, listener_service)
        self.on_end, self.pc, self.speaker, self.sideband = on_end, None, None, None
        self.inbox = asyncio.Queue(maxsize=512)  # Session.handle calls, in arrival order.
        self.pending, self.tasks = deque(), set()
        self.delegations = {}  # Open delegation id -> Frankie call id.
        self.offset, self.last_speaker = 0, None
        self.context, self.context_revision = [], 0
        self.muted = self.closed = self.attached = False

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def settings(self, session):
        frankie = {**FRANKIE, **(LISTENER if self.session.listener is not None else {})}
        rate = max(getattr(self.session.engine, "output_rates", (24000,)))
        return {"instructions": "\n\n".join(filter(None, (STYLE, session["instructions"].strip(), DELEGATE_GUIDE))),
                "output_modalities": ["audio"], "tools": [DELEGATE],
                "audio": {"output": {"format": {"type": "audio/pcm", "rate": rate}}},
                "frankie": frankie}

    async def configure(self, session):
        self.spawn(self.session.send_events())
        self.spawn(self.pump())
        await self.session.handle({"type": "session.update", "session": self.settings(session)})
        for item in session["items"]:
            await self.session.handle({"type": "conversation.item.create", "item": item})

    async def open(self, session, offer):
        """Configure Frankie and answer the offer; returns the answer SDP."""
        await self.configure(session)
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))

        @self.pc.on("track")
        def heard(track):
            if track.kind == "audio":
                self.spawn(self.listen(track))

        @self.pc.on("connectionstatechange")
        async def changed():
            if self.pc.connectionState in {"failed", "closed"}:
                await self.close("transport")

        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
        self.speaker = Speaker(self.session.settings["output_rate"], self.played)
        self.pc.addTrack(self.speaker)
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        self.spawn(self.expect_attach())
        return self.pc.localDescription.sdp

    def feed(self, event):
        try:
            self.inbox.put_nowait((event, None))
        except asyncio.QueueFull:
            pass  # Audio Frankie cannot keep up with is dropped, as a stalled network would.

    async def pump(self):
        while True:
            event, ack = await self.inbox.get()
            try:
                await self.session.handle(event)
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                if ack:
                    await self.error(ack[0], "invalid_request", str(exc))
                else:
                    print(f"Live session {self.id}: {type(exc).__name__}: {exc}", flush=True)
            else:
                if ack:
                    clock = int(self.session.clock_ms)
                    await self.emit(f"session.{ack[1]}.appended", client_event_id=ack[0],
                                    start_ms=clock, end_ms=clock)

    async def listen(self, track):
        resampler = AudioResampler(format="s16", layout="mono", rate=EAR_RATE)
        while not self.closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            for out in resampler.resample(frame):
                mic = out.to_ndarray().reshape(-1).astype("<i2")
                if self.muted:
                    mic = np.zeros_like(mic)
                self.feed({"type": "input_audio_buffer.append",
                            "audio": base64.b64encode(mic.tobytes()).decode(),
                            "playback": base64.b64encode(self.speaker.heard_for(len(mic)).tobytes()).decode()})

    def played(self, kind, item, ms):
        response_id, item_id = item
        if kind == "truncate":
            self.feed({"type": "conversation.item.truncate", "item_id": item_id,
                        "content_index": 0, "audio_end_ms": ms})
        else:
            self.feed({"type": "frankie.playback." + kind, "item_id": item_id,
                        "response_id": response_id, "audio_end_ms": ms})

    async def frankie_event(self, event):
        kind, speaker = event["type"], self.speaker
        if kind == "response.output_audio.delta" and speaker:
            speaker.audio(event["response_id"], event["item_id"], pcm(event["delta"]))
        elif kind == "response.output_audio.done" and speaker:
            speaker.done(event["item_id"])
        elif kind == "frankie.backchannel.delta" and speaker:
            speaker.backchannel(event["item_id"], pcm(event["delta"]))
        elif kind == "frankie.playback.clear" and speaker:
            speaker.clear(event.get("response_id"))
        elif kind in {"frankie.playback.pause", "frankie.playback.resume"} and speaker:
            speaker.pause(event["response_id"], kind.endswith("pause"))
        elif kind == "conversation.item.input_audio_transcription.completed":
            item = next((i for i in self.session.items if i["id"] == event["item_id"]), {})
            end = int(item.get("_speech_end_ms", self.session.clock_ms))
            await self.transcript("session.input_transcript.delta", ("user", item.get("id")),
                                  event["transcript"].strip(), min(int(item.get("_speech_start_ms", end)), end), end)
        elif kind == "response.output_audio_transcript.delta":
            clock = int(self.session.clock_ms)
            await self.transcript("session.output_transcript.delta", ("assistant", event["response_id"]),
                                  event["delta"], clock, clock)
        elif kind == "response.function_call_arguments.done" and event["name"] == DELEGATE["name"]:
            delegation_id = identifier("dlg")
            self.delegations[delegation_id] = event["call_id"]
            # Everything said so far on the session clock is the request's context.
            self.offset = max(self.offset, int(self.session.clock_ms))
            await self.emit("session.delegation.created", offset_ms=self.offset,
                            delegation={"id": delegation_id, "type": "delegation", "target": "client"})

    async def transcript(self, kind, speaker, text, start_ms, end_ms):
        if not text:
            return
        # The client joins a speaker's consecutive deltas; keep separate turns apart.
        if self.last_speaker and self.last_speaker[0] == speaker[0] and self.last_speaker != speaker:
            text = " " + text.lstrip()
        self.last_speaker = speaker
        await self.emit(kind, delta=text, start_ms=start_ms, end_ms=end_ms)

    async def command(self, message):
        kind = message.get("type") if isinstance(message, dict) else None
        event_id = message.get("event_id") if isinstance(message, dict) else None
        if not isinstance(event_id, str) or not event_id:
            return await self.error(None, "invalid_request", "Client events need an event_id.")
        if kind == "session.close":
            return await self.close("client", event_id)
        if kind in {"session.input_audio.mute", "session.input_audio.unmute"}:
            self.muted = kind.endswith(".mute")
            return await self.emit("session.input_audio." + ("muted" if self.muted else "unmuted"),
                                   client_event_id=event_id)
        channel = CONTEXT_CHANNELS.get(kind)
        if channel is None:
            return await self.error(event_id, "unsupported_event", f"Frankie does not accept {kind}.")
        content = message.get("content")
        if not isinstance(content, str) or len(content.encode()) > MAX_COMMENTARY_BYTES:
            return await self.error(event_id, "invalid_request", "Context content must be text under 64 KiB.")
        delegation_id = message.get("delegation_id")
        if channel == "commentary" and delegation_id is not None:
            # One result per delegation: an unknown, answered or closed id is refused, never spoken.
            call_id = self.delegations.pop(delegation_id, None)
            if call_id is None:
                return await self.error(event_id, "unknown_delegation", "Commentary needs an open delegation_id.")
            await self.inbox.put(({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": call_id, "output": content}}, (event_id, channel)))
            await self.inbox.put(({"type": "response.create"}, None))
            return
        if delegation_id is not None:
            return await self.error(event_id, "invalid_request", "Only commentary belongs to a delegation.")
        # Context, not a request: it rides along with the user's next turn, and nothing is spoken now.
        self.context.append(f"{CONTEXT_LABELS[channel]}: {content.strip()}")
        text = "\n".join(self.context)
        while len(text.encode()) > MAX_CONTEXT_BYTES and len(self.context) > 1:
            self.context.pop(0)
            text = "\n".join(self.context)
        self.context_revision += 1
        await self.inbox.put(({"type": "frankie.input_context.update", "revision": self.context_revision,
                               "text": text.encode()[:MAX_CONTEXT_BYTES].decode(errors="ignore")},
                              (event_id, channel)))

    async def emit(self, kind, **data):
        event = {"type": kind, "event_id": identifier("evt"), **data}
        if self.sideband is not None:
            try:
                await self.sideband.send_json(event)
            except Exception:
                self.spawn(self.close("sideband_closed"))
        elif len(self.pending) < MAX_PENDING_EVENTS:
            self.pending.append(event)
        else:
            self.spawn(self.close("sideband_backlog"))

    async def error(self, client_event_id, code, message):
        await self.emit("error", error={"type": "invalid_request_error", "code": code, "message": message,
                                        **({"client_event_id": client_event_id} if client_event_id else {})})

    async def attach(self, ws):
        if self.closed or self.attached:
            await ws.close(code=1008, reason="This live session already has a sideband.")
            return
        self.attached = True
        self.tasks.add(asyncio.current_task())  # Closing elsewhere ends this receive loop too.
        try:
            await ws.accept()
            # Events raised meanwhile queue behind these, so the client sees them in order.
            await ws.send_json({"type": "session.started", "event_id": identifier("evt"),
                                "session": {"id": self.id}})
            while self.pending:
                await ws.send_json(self.pending.popleft())
            self.sideband = ws
            while not self.closed:
                try:
                    message = await ws.receive_json()
                except (ValueError, KeyError):
                    await self.error(None, "invalid_request", "Send JSON client events.")
                    continue
                await self.command(message)
        except Exception:
            pass  # Disconnects end the session below.
        finally:
            await self.close("sideband_closed")

    async def expect_attach(self):
        await asyncio.sleep(ATTACH_TIMEOUT_S)
        if not self.attached:
            await self.close("attach_timeout")

    async def close(self, reason="server", client_event_id=None):
        if self.closed:
            return
        self.closed = True
        current = asyncio.current_task()
        try:
            if self.sideband is not None:
                try:
                    await self.sideband.send_json({
                        "type": "session.closed", "event_id": identifier("evt"), "reason": reason,
                        "session": {"id": self.id}, "usage": {},
                        **({"client_event_id": client_event_id} if client_event_id else {})})
                except Exception:
                    pass
            for task in tuple(self.tasks):
                if task is not current:
                    task.cancel()
            await self.session.close()
            if self.pc is not None:
                await self.pc.close()
            if self.sideband is not None:
                try:
                    await self.sideband.close()
                except Exception:
                    pass
        finally:
            self.on_end(self)


def attach_routes(app, token, seat, get_engine, executor, listener_service):
    """Serve live sessions; ``seat.owner`` is the one conversation Frankie holds."""
    from fastapi import Request, WebSocket
    from fastapi.responses import JSONResponse

    sessions = {}
    expected = ("Bearer " + token).encode()

    def authorized(header):
        return hmac.compare_digest(header.encode("utf-8", "replace"), expected)

    def fail(status, code, message):
        return JSONResponse({"error": {"type": "invalid_request_error", "code": code, "message": message}},
                            status_code=status)

    def ended(live):
        sessions.pop(live.id, None)
        if seat.owner is live:
            seat.owner = None

    @app.post("/v1/live/sessions")
    async def create(request: Request):
        if not authorized(request.headers.get("authorization", "")):
            return fail(401, "invalid_api_key", "Invalid access token.")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return fail(413, "too_large", "Session requests are limited to 1 MiB.")
        try:
            session, offer = parse(json.loads(body))
        except ValueError as exc:
            return fail(400, "invalid_request", str(exc))
        engine = get_engine()
        if engine is None:
            return fail(503, "loading", "Frankie is still loading.")
        if seat.owner is not None:
            return fail(409, "busy", "Another conversation owns Frankie.")
        live = LiveSession(engine, executor, listener_service(), ended)
        seat.owner = sessions[live.id] = live
        try:
            answer = await asyncio.wait_for(live.open(session, offer), SETUP_TIMEOUT_S)
        except asyncio.CancelledError:
            await asyncio.shield(live.close("setup_cancelled"))
            raise
        except Exception as exc:
            await live.close("setup_failed")
            if isinstance(exc, ValueError):
                return fail(400, "invalid_request", str(exc))
            return fail(500, "setup_failed", f"Live session setup failed: {type(exc).__name__}.")
        return JSONResponse({"session": {"id": live.id, "object": "live.session", "model": session["model"]},
                             "transport": {"type": "webrtc", "sdp": answer}}, status_code=201)

    @app.websocket("/v1/live/sessions/{session_id}/attach")
    async def attach(ws: WebSocket, session_id: str):
        live = sessions.get(session_id)
        if not authorized(ws.headers.get("authorization", "")) or live is None:
            await ws.close(code=1008, reason="Unknown live session or invalid access token.")
            return
        await live.attach(ws)
