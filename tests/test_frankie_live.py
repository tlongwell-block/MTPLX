import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("aiortc")
import numpy as np

from mtplx.frankie import live


class Engine:
    """Answers a typed request by calling delegate once, then speaks the delegated result."""

    tokenizer = None
    audio = NS(make_vad=lambda: NS(reset=lambda: None))
    output_rates = (24000, 48000)

    def __init__(self):
        self.calls = []

    def prepare_audio(self, item):
        return []

    def respond(self, history, settings, emit, abort, **kwargs):
        self.calls.append((history, settings))
        if history[-1].get("type") == "message" and history[-1]["role"] == "user" \
                and "Background task notice" not in json.dumps(history[-1]):
            emit("text", "On it.")
            emit("tool_call", {"key": [len(self.calls), 0],
                               "call": {"id": f"call_{len(self.calls)}",
                                        "function": {"name": "delegate", "arguments": "{}"}}})
        else:
            emit("text", "The hostname is studio.")
        return {"raw_text": "", "finish_reason": "stop", "stats": {}, "audio_seconds": 0, "seconds": 0.01}

    def warm(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass


class Socket:
    def __init__(self):
        self.sent, self.inbox, self.closed = [], asyncio.Queue(), False

    async def accept(self):
        pass

    async def send_json(self, event):
        self.sent.append(event)

    async def receive_json(self):
        return await self.inbox.get()

    async def close(self, code=1000, reason=""):
        self.closed = True

    def of(self, kind):
        return [e for e in self.sent if e["type"] == kind]


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


def run(check):
    async def main():
        with ThreadPoolExecutor(max_workers=1) as executor:
            engine, ended = Engine(), []
            s = live.LiveSession(engine, executor, None, ended.append)
            session, _ = live.parse(body())
            await s.configure(session)
            ws = Socket()
            attach = asyncio.create_task(s.attach(ws))
            await until(lambda: s.sideband is ws)
            try:
                await check(s, ws, engine)
            finally:
                await s.close()
                await asyncio.wait_for(asyncio.gather(attach, return_exceptions=True), 3)
            assert ended == [s]
    asyncio.run(main())


def body(**session):
    return {"session": {"model": "frankie-v10", "instructions": "Help.", "delegation": {"type": "client"},
                        "input": [{"type": "message", "role": "user",
                                   "content": [{"type": "input_text", "text": "earlier"}]}], **session},
            "transport": {"type": "webrtc", "sdp": "v=0\r\n"}}


async def ask(s, text="What is this computer called?"):
    await s.session.handle({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
    await s.session.handle({"type": "response.create"})


def test_parse_accepts_goose_and_refuses_what_it_cannot_serve():
    session, sdp = live.parse(body())
    assert sdp == "v=0\r\n" and session["model"] == "frankie-v10"
    assert session["items"][0]["content"][0]["text"] == "earlier"
    for bad in (body(delegation={"type": "responses"}), body(model=""), body(input="x"),
                body(input=[{"type": "message", "role": "tool", "content": []}]),
                {"session": {"model": "m"}, "transport": {"type": "websocket", "sdp": "x"}},
                {"session": {"model": "m"}, "transport": {"type": "webrtc", "sdp": "x" * 70000}}, []):
        with pytest.raises(ValueError):
            live.parse(bad)


def test_session_starts_first_and_configures_frankie_for_delegation():
    async def check(s, ws, engine):
        await until(lambda: ws.sent)
        assert ws.sent[0]["type"] == "session.started" and ws.sent[0]["session"]["id"] == s.id
        settings = s.session.settings
        assert settings["tools"] == [live.DELEGATE] and settings["background_tasks"]
        assert settings["output_rate"] == 48000 and settings["playback_feedback"]
        assert "Help." in settings["instructions"] and live.DELEGATE_GUIDE in settings["instructions"]
        assert s.session.items[0]["content"][0]["text"] == "earlier"
    run(check)


def test_delegation_result_is_acknowledged_and_spoken_in_frankies_words():
    async def check(s, ws, engine):
        await ask(s)
        await until(lambda: ws.of("session.delegation.created"))
        created = ws.of("session.delegation.created")[0]
        delegation = created["delegation"]
        assert delegation["type"] == "delegation" and delegation["target"] == "client"
        assert isinstance(created["offset_ms"], int)
        await ws.inbox.put({"type": "session.commentary.append", "event_id": "evt_c1",
                            "delegation_id": delegation["id"], "content": "**studio**"})
        await until(lambda: ws.of("session.commentary.appended"))
        assert ws.of("session.commentary.appended")[0]["client_event_id"] == "evt_c1"
        await until(lambda: len(engine.calls) == 2)
        history = json.dumps(engine.calls[1][0])
        assert "**studio**" in history and "ready_to_report" in history  # Tool data for the brain.
        await until(lambda: any("hostname" in e["delta"] for e in ws.of("session.output_transcript.delta")))
    run(check)


def test_each_delegation_takes_one_result_and_unknown_ids_are_refused():
    async def check(s, ws, engine):
        await ask(s)
        await until(lambda: ws.of("session.delegation.created"))
        did = ws.of("session.delegation.created")[0]["delegation"]["id"]
        for event_id, delegation_id in (("evt_a", did), ("evt_b", did), ("evt_c", "dlg_unknown")):
            await ws.inbox.put({"type": "session.commentary.append", "event_id": event_id,
                                "delegation_id": delegation_id, "content": "done"})
        await until(lambda: len(ws.of("error")) == 2)
        assert [e["client_event_id"] for e in ws.of("session.commentary.appended")] == ["evt_a"]
        errors = {e["error"]["client_event_id"]: e["error"]["code"] for e in ws.of("error")}
        assert errors == {"evt_b": "unknown_delegation", "evt_c": "unknown_delegation"}
    run(check)


def test_delegation_offsets_cover_the_request_transcript_and_never_move_back():
    async def check(s, ws, engine):
        s.session.clock_ms = 3000
        item = {"id": "item_u", "type": "message", "role": "user", "content": [],
                "_speech_start_ms": 900, "_speech_end_ms": 2800}
        s.session.items.append(item)
        await s.frankie_event({"type": "conversation.item.input_audio_transcription.completed",
                               "item_id": "item_u", "content_index": 0, "transcript": " run hostname "})
        heard = ws.of("session.input_transcript.delta")[-1]
        assert (heard["delta"], heard["start_ms"], heard["end_ms"]) == ("run hostname", 900, 2800)
        await s.frankie_event({"type": "response.function_call_arguments.done", "name": "delegate",
                               "call_id": "call_x"})
        s.session.clock_ms = 1000  # A clock reading can never move a delegation earlier.
        await s.frankie_event({"type": "response.function_call_arguments.done", "name": "delegate",
                               "call_id": "call_y"})
        first, second = ws.of("session.delegation.created")[-2:]
        assert first["offset_ms"] == second["offset_ms"] == 3000 >= heard["end_ms"]
        assert first["delegation"]["id"] != second["delegation"]["id"]
    run(check)


def test_separate_user_turns_stay_separate_words():
    async def check(s, ws, engine):
        for item_id, text in (("u1", "hello"), ("u2", "again")):
            s.session.items.append({"id": item_id, "type": "message", "role": "user", "content": []})
            await s.frankie_event({"type": "conversation.item.input_audio_transcription.completed",
                                   "item_id": item_id, "content_index": 0, "transcript": text})
        assert [e["delta"] for e in ws.of("session.input_transcript.delta")] == ["hello", " again"]
    run(check)


def test_mute_close_and_bad_events_answer_their_client_event():
    async def check(s, ws, engine):
        await ws.inbox.put({"type": "session.input_audio.mute", "event_id": "m1"})
        await ws.inbox.put({"type": "session.input_audio.append", "event_id": "t1", "audio": "AAAA"})
        await until(lambda: ws.of("error"))
        assert ws.of("session.input_audio.muted")[0]["client_event_id"] == "m1" and s.muted
        assert ws.of("error")[0]["error"]["client_event_id"] == "t1"
        await ws.inbox.put({"type": "session.close", "event_id": "c1"})
        await until(lambda: s.closed)
        closed = ws.of("session.closed")[0]
        assert closed["client_event_id"] == "c1" and closed["reason"] == "client"
        assert closed["session"] == {"id": s.id} and ws.closed
    run(check)


def frames(speaker, n):
    return [speaker.next_frame() for _ in range(n)]


def test_speaker_holds_a_preroll_reports_playback_and_never_truncates_a_heard_reply():
    reports = []
    speaker = live.Speaker(48000, lambda kind, item, ms: reports.append((kind, ms)))
    speaker.audio("r1", "i1", np.ones(48 * 100, np.int16))
    assert not frames(speaker, 1)[0].any()  # 100 ms is under the 120 ms preroll.
    speaker.audio("r1", "i1", np.ones(48 * 100, np.int16))
    speaker.done("i1")
    out = frames(speaker, 12)
    assert all(f.all() for f in out[:10]) and not out[10].any()
    assert reports[-1] == ("finished", 200) and ("truncate", 200) not in reports
    assert len(speaker.heard_for(480)) == 480  # The ear gets a 24 kHz echo reference.


def test_speaker_barge_in_truncates_at_what_was_heard_and_drops_the_rest():
    reports = []
    speaker = live.Speaker(24000, lambda kind, item, ms: reports.append((kind, item, ms)))
    speaker.audio("r1", "i1", np.ones(24 * 1000, np.int16))
    frames(speaker, 5)
    speaker.clear("r1")
    assert reports[-1] == ("truncate", ("r1", "i1"), 100)
    speaker.audio("r1", "i1", np.ones(2400, np.int16))  # Late audio of a cleared reply stays silent.
    assert not any(f.any() for f in frames(speaker, 10))


def test_speaker_plays_listening_sounds_only_between_replies():
    speaker = live.Speaker(24000, lambda *a: None)
    speaker.backchannel("bc1", np.full(480, 7, np.int16))
    assert (frames(speaker, 1)[0] == 7).all()
    speaker.audio("r1", "i1", np.ones(24 * 200, np.int16))
    speaker.backchannel("bc2", np.full(480, 7, np.int16))
    assert not any((f == 7).any() for f in frames(speaker, 12))


def test_a_sideband_that_fails_to_accept_still_frees_the_seat():
    async def main():
        with ThreadPoolExecutor(max_workers=1) as executor:
            ended = []
            s = live.LiveSession(Engine(), executor, None, ended.append)
            await s.configure(live.parse(body())[0])
            ws = Socket()

            async def broken():
                raise RuntimeError("handshake failed")
            ws.accept = broken
            await s.attach(ws)
            assert s.closed and ended == [s]
    asyncio.run(main())


def test_instructions_and_thinking_are_acknowledged_context_for_the_next_turn():
    async def check(s, ws, engine):
        await ws.inbox.put({"type": "session.instructions.append", "event_id": "i1", "content": "Be brief."})
        await ws.inbox.put({"type": "session.thinking.append", "event_id": "t1", "content": "Checking disk."})
        await ws.inbox.put({"type": "session.instructions.append", "event_id": "i2", "delegation_id": "dlg_x",
                            "content": "x"})
        await until(lambda: ws.of("session.thinking.appended") and ws.of("error"))
        assert ws.of("session.instructions.appended")[0]["client_event_id"] == "i1"
        assert ws.of("session.thinking.appended")[0]["client_event_id"] == "t1"
        assert ws.of("error")[0]["error"]["client_event_id"] == "i2"
        revision, text = s.session.latest_context
        assert revision == 2 and "Be brief." in text and "Checking disk." in text
        assert not ws.of("session.output_transcript.delta") and not engine.calls  # Nothing is spoken.
    run(check)


def test_a_cancelled_setup_frees_the_seat():
    async def main():
        from fastapi import FastAPI
        app, seat, engine = FastAPI(), NS(owner=None), Engine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            live.attach_routes(app, "tok", seat, lambda: engine, executor, lambda: None)
            create = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/v1/live/sessions")
            opened = asyncio.Event()

            async def stalled(self, session, offer):
                opened.set()
                await asyncio.sleep(60)
            original, live.LiveSession.open = live.LiveSession.open, stalled
            try:
                request = NS(headers={"authorization": "Bearer tok"},
                             body=lambda: asyncio.sleep(0, json.dumps(body()).encode()))
                task = asyncio.create_task(create(request))
                await opened.wait()
                assert seat.owner is not None
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert seat.owner is None
            finally:
                live.LiveSession.open = original
    asyncio.run(main())


def test_the_guide_names_every_way_a_client_asks_for_delegation():
    # Live clients say "delegate to the client" or "to Goose"; either must mean the tool, not a spoken promise.
    for word in ("Goose", "the client", "calling delegate", "does not delegate anything"):
        assert word in live.DELEGATE_GUIDE
