import asyncio
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("soundfile")
pytest.importorskip("scipy")
import numpy as np

from mtplx.frankie.session import Session, public


class Engine:
    tokenizer = None
    audio = NS(make_vad=lambda: NS(reset=lambda: None))

    def __init__(self):
        self.calls = []
        self.releases = [Event() for _ in range(10)]

    def prepare_audio(self, item):
        return [
            (item, i, "How is it going?")
            for i, p in enumerate(item.get("content", []))
            if p["type"] == "input_audio"
        ]

    def respond(self, history, settings, emit, abort, **kwargs):
        index = len(self.calls)
        # Transcription can publish cached fields while deepcopy copies PCM.
        # Snapshot small containers first; do not change production cache sharing.
        snapshot = []
        for item in history:
            item = dict(item)
            if "content" in item:
                item["content"] = [dict(part) for part in item["content"]]
            snapshot.append(item)
        self.calls.append(copy.deepcopy(snapshot))
        emit("text", "I am listening.")
        while not self.releases[index].wait(0.002) and not abort.is_set():
            pass
        return {
            "raw_text": "I am listening.",
            "finish_reason": "stop",
            "stats": {},
            "audio_seconds": 0,
            "seconds": 0.01,
        }

    def warm(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass


async def wait_for(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.002)


async def setup(check):
    with ThreadPoolExecutor(max_workers=1) as executor:
        engine = Engine()
        session = Session(engine, executor, NS())
        try:
            await session.handle(
                {
                    "type": "session.update",
                    "session": {
                        "frankie": {"background_tasks": True},
                        "output_modalities": ["text"],
                    },
                }
            )
            await check(session, engine)
        finally:
            for release in engine.releases:
                release.set()
            await session.close()


async def call(s, call_id="lookup_1"):
    await s.handle(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call",
                "call_id": call_id,
                "name": "lookup",
                "arguments": "{}",
            },
        }
    )


async def output(s, call_id="lookup_1", value="42"):
    await s.handle(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": value,
            },
        }
    )


async def user(s, text="Keep talking while that runs."):
    await s.handle(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
    )


def test_pending_call_does_not_block_user_and_active_reply_gets_immutable_snapshot():
    async def check(s, e):
        await call(s)
        await user(s)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        first = copy.deepcopy(e.calls[0])
        assert e.calls[0][1]["type"] == "function_call_output"
        assert '"pending"' in e.calls[0][1]["output"]
        await output(s)
        assert e.calls[0] == first and not s.current.abort.is_set()
        await s.handle({"type": "response.create"})
        await s.handle(
            {"type": "response.create"}
        )  # Coalesce duplicate result triggers.
        assert s.queued_task_response
        e.releases[0].set()
        await wait_for(lambda: len(e.calls) == 2)
        texts = [p.get("text", "") for i in e.calls[1] for p in i.get("content", [])]
        assert texts.index("Keep talking while that runs.") < next(
            i for i, text in enumerate(texts) if "Background task notice" in text
        )
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        await s.handle(
            {"type": "response.create"}
        )  # Already consumed result cannot speak twice.
        assert len(e.calls) == 2
        assert s.task_ledger.tasks["lookup_1"].status == "completed"

    asyncio.run(setup(check))


@pytest.mark.parametrize("reason", ["cancelled", "superseded"])
def test_late_obsolete_results_and_stale_response_create_never_speak(reason):
    async def check(s, e):
        await call(s)
        await user(s)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        await s.handle(
            {"type": "frankie.task.cancel", "call_id": "lookup_1", "reason": reason}
        )
        await output(s, value="OBSOLETE_SECRET")
        await s.handle({"type": "response.create"})
        assert not s.queued_task_response and not s.current.abort.is_set()
        e.releases[0].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        assert len(e.calls) == 1
        await user(s, "New request.")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 2)
        assert "OBSOLETE_SECRET" not in str(e.calls[-1])
        assert "Cancellation" not in str(e.calls[0])
        with pytest.raises(ValueError, match="already supplied"):
            await output(s)

    asyncio.run(setup(check))


def test_typed_input_preempts_queued_result_reply_and_results_remain_in_next_snapshot():
    async def check(s, e):
        await call(s)
        await user(s)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        await output(s)
        await s.handle({"type": "response.create"})
        previous = s.current
        await user(s, "Actually, summarize briefly.")
        assert previous.abort.is_set() and s.queued_task_response
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 2)
        assert e.calls[-1][-1]["content"][0]["text"] == "Actually, summarize briefly."
        assert "Background task notice" in str(e.calls[-1])
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        assert len(e.calls) == 2

    asyncio.run(setup(check))


def test_manual_spoken_input_works_while_tool_is_pending():
    async def check(s, e):
        await call(s)
        s.settings["turn_detection"] = None
        audio = base64.b64encode(np.zeros(2400, dtype="<i2")).decode()
        await s.receive_audio(audio)
        await s.handle({"type": "input_audio_buffer.commit"})
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        assert e.calls[0][-1]["content"][0]["type"] == "input_audio"
        assert '"pending"' in e.calls[0][1]["output"]

    asyncio.run(setup(check))


def test_playback_feedback_requires_valid_monotonic_actual_drain_and_retains_old_run():
    async def check(s, e):
        await user(s)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        run = s.current
        run.emitted_ms = 1000
        event = {
            "type": "frankie.playback.position",
            "item_id": run.item_id,
            "response_id": run.id,
            "audio_end_ms": 400,
        }
        await s.handle(event)
        assert run.played_ms == 400
        for bad in (
            {"audio_end_ms": 399},
            {"audio_end_ms": 1002},
            {"response_id": "stale"},
        ):
            with pytest.raises(ValueError, match="Invalid or stale"):
                await s.handle({**event, **bad})
        with pytest.raises(ValueError, match="before response completion"):
            await s.handle(
                {**event, "type": "frankie.playback.finished", "audio_end_ms": 1000}
            )
        e.releases[0].set()
        await wait_for(lambda: run.done)
        await user(s, "Next turn.")
        await s.handle({"type": "response.create"})
        await s.handle(
            {**event, "type": "frankie.playback.finished", "audio_end_ms": 1000}
        )
        assert run.playback_finished and s.current is not run
        await s.handle(
            {
                "type": "conversation.item.truncate",
                "item_id": run.item_id,
                "audio_end_ms": 1000,
            }
        )
        assert "interrupted" not in str(public(run.item))
        assert not s.current.abort.is_set()

    asyncio.run(setup(check))


def test_superseding_completed_unspoken_result_removes_queued_reply_without_undo_claim():
    async def check(s, e):
        await call(s)
        await user(s)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        await output(s, value="OLD_INFORMATION")
        await s.handle({"type": "response.create"})
        assert s.queued_task_response
        await s.handle(
            {
                "type": "frankie.task.cancel",
                "call_id": "lookup_1",
                "reason": "superseded",
            }
        )
        assert not s.queued_task_response and not s.unhandled_task_results
        events = []
        while not s.outgoing.empty():
            events.append(s.outgoing.get_nowait())
        cancellation = [
            event for event in events if event["type"] == "frankie.task.updated"
        ][-1]
        assert cancellation["task"]["status"] == "superseded"
        assert cancellation["cancellation_requested"] is False
        e.releases[0].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        await asyncio.sleep(0.01)
        assert len(e.calls) == 1

    asyncio.run(setup(check))
