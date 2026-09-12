import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import base64
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("soundfile")
pytest.importorskip("scipy")
import numpy as np
from mtplx.frankie.session import Session, Response, public


class Vad:
    def reset(self):
        pass

    def __call__(self, frame):
        return float(np.max(np.abs(frame)) > 0.1)


class Engine:
    audio = NS(make_vad=lambda: Vad())
    bank = NS(clear=lambda **kw: None)

    def respond(self, items, settings, emit, abort, **kwargs):
        emit("text", "Hello.")
        abort.wait(0.02)
        return {
            "text": "Hello.",
            "raw_text": "Hello.",
            "stats": {},
            "finish_reason": "stop",
            "audio_seconds": 0,
            "chunks": [],
            "seconds": 0.02,
        }

    tokenizer = None

    def warm(self, *a, **k):
        pass

    def release(self, *a, **k):
        pass


async def setup(fn):
    with ThreadPoolExecutor(max_workers=1) as executor:
        session = Session(Engine(), executor, NS())
        try:
            await fn(session)
        finally:
            await session.close()


def test_tentative_response_is_never_published_after_resume():
    async def check(s):
        s.start(s.audio_item(np.zeros(2400)), tentative=True)
        await asyncio.sleep(0.03)
        s.discard_spec()
        await asyncio.gather(*s.tasks)
        assert not s.items
        types = []
        while not s.outgoing.empty():
            types.append(s.outgoing.get_nowait()["type"])
        assert "frankie.speculation.aborted" in types
        assert "response.created" not in types
        assert "response.output_audio_transcript.delta" not in types

    asyncio.run(setup(check))


def test_truncate_preserves_only_completed_heard_chunks():
    async def check(s):
        run = Response(
            visible=True,
            settings=s.settings,
            item={
                "id": "answer",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "audio", "transcript": "First. Never heard."}],
            },
        )
        run.item_id = "answer"
        run.chunks = [
            {"text": "First.", "end_ms": 500},
            {"text": "Never heard.", "end_ms": 1200},
        ]
        s.current = run
        s.items = [run.item]
        await s.handle(
            {
                "type": "conversation.item.truncate",
                "item_id": "answer",
                "audio_end_ms": 600,
            }
        )
        assert (
            run.item["content"][0]["transcript"] == "First. [interrupted by the user]"
        )
        assert run.abort.is_set()

    asyncio.run(setup(check))


def test_settings_cannot_drop_pending_pcm():
    async def check(s):
        s.settings["turn_detection"] = None
        await s.receive_audio(base64.b64encode(np.zeros(480, dtype="<i2")).decode())
        with pytest.raises(ValueError, match="pending audio"):
            await s.handle({"type": "session.update", "session": {"thinking": "low"}})
        assert s.settings["thinking"] == "off"
        await s.handle({"type": "input_audio_buffer.clear"})
        await s.handle({"type": "session.update", "session": {"thinking": "low"}})
        assert s.settings["thinking"] == "low"

    asyncio.run(setup(check))


def test_turn_gate_holds_pause_allows_resume_and_forces_bounded_release():
    from mtplx.frankie.turn import TurnWorker

    class Turn:
        failed = False
        latest = None
        samples = 0
        release = TurnWorker.release

        def append(self, user, system):
            self.samples += len(user)
            self.latest = (self.samples, 0.1)

        def close(self):
            pass

    async def check(s):
        s.turn = Turn()

        async def feed(value, frames):
            pcm = base64.b64encode(np.full(768, value, dtype="<i2")).decode()
            for _ in range(frames):
                await s.receive_audio(pcm)
                await asyncio.sleep(0)

        await feed(16000, 4)
        await feed(0, 12)
        assert s.spec is not None and not s.spec.visible
        assert s.listening
        await feed(16000, 3)
        assert s.spec is None
        assert s.metrics["speculation_aborts"] == 1
        await feed(0, 48)
        assert not s.listening
        assert s.current.visible

    asyncio.run(setup(check))


def test_playback_is_aligned_and_cleared_with_capture():
    async def check(s):
        mic = base64.b64encode(np.full(480, 1000, dtype="<i2")).decode()
        played = base64.b64encode(np.full(480, 2000, dtype="<i2")).decode()
        with pytest.raises(ValueError, match="align exactly"):
            await s.receive_audio(mic, "")
        assert s.received_ms == 0
        await s.receive_audio(mic, played)
        assert len(s.tail) == len(s.system_tail) == 480
        np.testing.assert_allclose(s.system_tail, 2 * s.tail)
        await s.handle({"type": "input_audio_buffer.clear"})
        assert len(s.tail) == len(s.system_tail) == 0
        await s.receive_audio(mic)
        np.testing.assert_array_equal(s.system_tail, 0)

    asyncio.run(setup(check))


def test_duplicate_tool_result_is_rejected():
    async def check(s):
        s.items = [{"id": "call", "type": "function_call", "call_id": "lookup"}]
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "lookup",
                "output": "42",
            },
        }
        await s.handle(event)
        with pytest.raises(ValueError, match="already supplied"):
            await s.handle(event)

    asyncio.run(setup(check))


def test_harness_can_restore_tool_call_history():
    async def check(s):
        call = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call",
                "name": "lookup",
                "call_id": "lookup-1",
                "arguments": '{"key":"test"}',
            },
        }
        await s.handle(call)
        await s.handle(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": "lookup-1",
                    "output": "42",
                },
            }
        )
        assert len(s.items) == 2
        call["item"]["id"] = "different-item"
        with pytest.raises(ValueError, match="Duplicate tool call"):
            await s.handle(call)

    asyncio.run(setup(check))


def test_vad_item_id_and_timeline_survive_manual_history_import():
    async def check(s):
        # History replay does not advance the live microphone ledger.
        s.settings["turn_detection"] = None

        def encoded(x):
            return base64.b64encode(np.full(768, x, dtype="<i2")).decode()

        await s.receive_audio(encoded(0))
        await s.handle({"type": "input_audio_buffer.commit"})
        assert s.received_ms == 0
        s.settings["turn_detection"] = {
            "create_response": False,
            "silence_duration_ms": 160,
        }
        await s.receive_audio(encoded(10000))
        for _ in range(5):
            await s.receive_audio(encoded(0))
        events = []
        while not s.outgoing.empty():
            events.append(s.outgoing.get_nowait())
        start = next(
            e for e in events if e["type"] == "input_audio_buffer.speech_started"
        )
        end = next(
            e for e in events if e["type"] == "input_audio_buffer.speech_stopped"
        )
        committed = [e for e in events if e["type"] == "input_audio_buffer.committed"][
            -1
        ]
        assert start["item_id"] == end["item_id"] == committed["item_id"]
        assert start["audio_start_ms"] == 0 and end["audio_end_ms"] == 192

    asyncio.run(setup(check))


def test_public_items_omit_runtime_arrays():
    assert public(
        {"content": [{"type": "input_audio", "_pcm": np.zeros(2), "_rows": object()}]}
    ) == {"content": [{"type": "input_audio"}]}
