import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("soundfile")
pytest.importorskip("scipy")
import numpy as np

from mtplx.frankie.session import Response, Session, public


class Vad:
    def reset(self):
        pass

    def __call__(self, frame):
        return float(np.max(np.abs(frame)) > 0.1)


class Engine:
    audio = NS(make_vad=lambda: Vad())
    bank = NS(clear=lambda **kw: None)

    def prepare_audio(self, item):
        return [
            (item, index, "Hello Frankie.")
            for index, part in enumerate(item.get("content", []))
            if part["type"] == "input_audio"
        ]

    def respond(self, items, settings, emit, abort, **kwargs):
        for item in items:
            for transcript in self.prepare_audio(item):
                emit("input_transcript", transcript)
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
        assert "conversation.item.input_audio_transcription.completed" not in types

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


@pytest.mark.parametrize("barrier", [None, "pause", "tool", "heard"])
def test_short_resume_survives_client_cancel_without_crossing_turns(barrier):
    import time

    async def check(s):
        original = s.audio_item(np.full(1024, 0.3))
        reply = {"id": "reply", "type": "message", "role": "assistant", "content": []}
        run = Response(
            input=original,
            item=reply,
            item_id="reply",
            visible=True,
            committed_at=time.monotonic(),
            speech_end_ms=100,
        )
        s.current, s.items = run, [original, reply]
        s.settings["turn_detection"]["interrupt_response"] = False
        s.received_ms = s.clock_ms = 500
        if barrier == "pause":
            run.speech_end_ms = -500
        if barrier == "tool":
            s.items.append({"id": "call", "type": "function_call"})
        if barrier == "heard":
            run.played_ms = 750
        s.cancel()
        run.done = True  # The client's cancellation can finish before frame three.
        frame = base64.b64encode(np.full(768, 16000, dtype="<i2")).decode()
        for _ in range(4):
            await s.receive_audio(frame)
        events = []
        while not s.outgoing.empty():
            events.append(s.outgoing.get_nowait())
        removed = {
            e["item_id"] for e in events if e["type"] == "conversation.item.deleted"
        }
        if barrier is None:
            assert run.merged and not s.items
            assert len(s.frames) == 5  # Original PCM is prepended exactly once.
            np.testing.assert_array_equal(s.frames[0], original["content"][0]["_pcm"])
            assert removed == {original["id"], "reply"}
            await s.handle(
                {
                    "type": "conversation.item.truncate",
                    "item_id": "reply",
                    "audio_end_ms": 0,
                }
            )
        else:
            assert not run.merged and original in s.items and not removed
            assert len(s.frames) == 4

    asyncio.run(setup(check))


def test_client_response_retains_committed_vad_input_for_resume():
    async def check(s):
        s.settings["turn_detection"]["create_response"] = False
        s.settings["turn_detection"]["interrupt_response"] = False
        speech = base64.b64encode(np.full(768, 16000, dtype="<i2")).decode()
        silence = base64.b64encode(np.zeros(768, dtype="<i2")).decode()
        for frame in [speech] * 4 + [silence] * 10:
            await s.receive_audio(frame)
        assert s.current is None and len(s.items) == 1
        item = s.items[0]
        await s.handle({"type": "response.create"})
        assert s.current.input is item
        assert s.current.speech_end_ms == 128
        assert "_speech_end_ms" not in public(item)
        s.cancel()

    asyncio.run(setup(check))


def test_duplex_turn_worker_keeps_both_heads_and_reset_fence(monkeypatch):
    import time

    from mtplx.frankie import turn

    class Model:
        def __init__(self, weights, mode):
            assert mode == "duplex"

        def process(self, user, system):
            return np.array([0.9, 0.1, 0.8])

        def reset(self):
            pass

    monkeypatch.setattr(turn, "TurnModel", Model)
    worker = turn.TurnWorker({}, mode="duplex")
    try:
        worker.append(np.zeros(1600), np.zeros(1600))
        deadline = time.monotonic() + 2
        while worker.latest is None and time.monotonic() < deadline:
            time.sleep(0.001)
        assert worker.latest == (1600, 0.1, 0.8)
        assert not worker.release(100, 320)
        assert worker.release(701, 320)
        worker.reset(1000)
        assert worker.latest is None
        assert worker.release(1000, 320)
    finally:
        worker.close()
    assert not worker.thread.is_alive()


def test_duplex_turn_frozen_reference():
    import os
    from pathlib import Path

    import mlx.core as mx
    from gguf import GGUFReader

    from mtplx.frankie.turn import TurnModel

    root = os.environ.get("FRANKIE_TURN_FIXTURE")
    if not root:
        pytest.skip("Set FRANKIE_TURN_FIXTURE to the frozen VAP-BC reference.")
    root = Path(root)
    weights = {}
    for tensor in GGUFReader(root / "duplex.gguf").tensors:
        value = tensor.data
        if value.ndim == 3:
            value = value.transpose(0, 2, 1)
        weights[tensor.name.removeprefix("turn.")] = mx.array(value.copy())
    model = TurnModel(weights, mode="duplex")
    audio = np.fromfile(root / "input.f32", np.float32).reshape(-1, 2, 1600)
    expected = np.fromfile(root / "duplex-output.f32", np.float32).reshape(-1, 3)
    actual = np.array([model.process(*frame) for frame in audio])
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)
    model.reset()
    np.testing.assert_allclose(model.process(*audio[0]), actual[0], atol=1e-6, rtol=0)


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


def drain(s):
    events = []
    while not s.outgoing.empty():
        events.append(s.outgoing.get_nowait())
    return events


@pytest.mark.parametrize("modality", ["audio", "text"])
def test_completed_response_uses_realtime_output_parts(modality):
    async def check(s):
        s.settings["output_modalities"] = [modality]
        s.start(s.audio_item(np.zeros(2400)))
        await asyncio.gather(*s.tasks)
        events = drain(s)
        response = next(e["response"] for e in events if e["type"] == "response.done")
        part = response["output"][0]["content"][0]
        assert response["status"] == "completed"
        assert part == {
            "type": "output_" + modality,
            "transcript" if modality == "audio" else "text": "Hello.",
        }
        finished = next(e for e in events if e["type"] == "response.content_part.done")
        assert finished["part"] == part
        input_event = next(
            e
            for e in events
            if e["type"] == "conversation.item.input_audio_transcription.completed"
        )
        assert input_event["item_id"] == s.items[0]["id"]
        assert input_event["content_index"] == 0
        assert input_event["transcript"] == "Hello Frankie."
        # Replayed audio history must not create another input transcription.
        s.start()
        await asyncio.gather(*s.tasks)
        assert not any("input_audio_transcription" in e["type"] for e in drain(s))

    asyncio.run(setup(check))


def test_speculative_transcript_waits_for_commit_without_stalling_worker():
    async def check(s):
        item = s.audio_item(np.zeros(2400))
        run = s.start(item, tentative=True)
        for _ in range(100):
            if run.transcripts:
                break
            await asyncio.sleep(0.002)
        assert run.transcripts
        assert not any("input_audio_transcription" in e["type"] for e in drain(s))
        assert "transcript" not in public(item)["content"][0]
        s.show(run)
        await asyncio.gather(*s.tasks)
        events = drain(s)
        types = [e["type"] for e in events]
        assert types.index("input_audio_buffer.committed") < types.index(
            "conversation.item.input_audio_transcription.completed"
        )
        assert item["content"][0]["transcript"] == "Hello Frankie."

    asyncio.run(setup(check))


def test_cancelled_and_removed_inputs_cannot_publish_late_transcripts():
    async def check(s):
        item = s.audio_item(np.zeros(2400))
        run = Response(input=item, settings=s.settings)
        s.show(run)
        drain(s)
        value = (item, 0, "Discarded speech.")
        run.abort.set()
        s.publish(run, "input_transcript", value)
        assert not drain(s)
        run.abort.clear()
        # A resumed utterance removed this item; even a queued callback cannot
        # attach the old transcription to a new item with the same identifier.
        s.items = [dict(item)]
        s.publish(run, "input_transcript", value)
        assert not drain(s)
        assert "transcript" not in item["content"][0]

    asyncio.run(setup(check))


def test_manual_commit_transcribes_without_creating_response():
    async def check(s):
        s.settings["turn_detection"] = None
        await s.receive_audio(base64.b64encode(np.zeros(2400, dtype="<i2")).decode())
        await s.handle({"type": "input_audio_buffer.commit"})
        await asyncio.gather(*s.tasks)
        types = [e["type"] for e in drain(s)]
        assert "conversation.item.input_audio_transcription.completed" in types
        assert "response.created" not in types

    asyncio.run(setup(check))
