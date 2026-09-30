"""A delayed result gets an explicit delivery cue without moving its knowledge."""

import asyncio
import json

import numpy as np
import pytest
from test_frankie_background_session import call, output, setup, user, wait_for
from test_frankie_semantic_session import feed


def notices(items):
    return [json.loads(part["text"].split("\n", 1)[1])
            for item in items for part in item.get("content", [])
            if part.get("text", "").startswith("Background task notice")]


def due(items):
    return [value for value in notices(items) if value.get("delivery") == "ready_to_report"]


def test_audio_turn_reserved_before_result_gets_its_answer_then_one_result_delivery_cue():
    async def check(session, engine):
        session.vad = lambda frame: float(np.max(np.abs(frame)) > 0.1)
        await user(session, "Check the inventory.")
        await call(session)
        await feed(session, 16000, 4)
        await output(session, value='{"available_units":13,"dispatch_day":"Thursday"}')
        await session.handle({"type": "response.create"})
        assert session.queued_task_response and session.listening
        await feed(session, 0, 10)
        await wait_for(lambda: len(engine.calls) == 1 and session.current.visible)
        assert not notices(engine.calls[0])  # Captured before the result arrived.
        assert session.unhandled_task_results == {"lookup_1"}
        engine.releases[0].set()
        await wait_for(lambda: len(engine.calls) == 2)
        history = engine.calls[1]
        results = [value for value in notices(history) if "output" in value]
        assert len(results) == 1
        assert json.loads(results[0]["output"]) == {"available_units": 13, "dispatch_day": "Thursday"}
        assert due(history) == [{"call_id": "lookup_1", "name": "lookup", "status": "completed",
                                 "delivery": "ready_to_report"}]
        assert due(history[-1:]) == due(history)
        result_at = next(i for i, item in enumerate(history) if notices([item]) and
                         "output" in notices([item])[0])
        audio_at = next(i for i, item in enumerate(history) if any(
            part["type"] == "input_audio" for part in item.get("content", [])))
        assert result_at < audio_at < len(history) - 2
        assert history[-2]["role"] == "assistant"
        assert not due(session.items)  # The cue is not authoritative user speech/history.
        assert not session.unhandled_task_results
        engine.releases[1].set()
        await wait_for(lambda: session.current.done)
        await session.handle({"type": "response.create"})
        assert len(engine.calls) == 2

    asyncio.run(setup(check))


def test_new_user_input_consumes_available_result_without_proactive_delivery_cue():
    async def check(session, engine):
        await user(session, "Check the inventory.")
        await call(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        await output(session, value="AVAILABLE_RESULT")
        await session.handle({"type": "response.create"})
        await user(session, "What did the lookup find? Explain it briefly.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2)
        assert "AVAILABLE_RESULT" in str(engine.calls[1])
        assert not due(engine.calls[1])
        assert engine.calls[1][-1]["content"][0]["text"] == "What did the lookup find? Explain it briefly."
        engine.releases[1].set()
        await wait_for(lambda: session.current.done)
        await session.handle({"type": "response.create"})
        assert len(engine.calls) == 2

    asyncio.run(setup(check))


def test_result_delivery_selects_only_current_unhandled_ids():
    async def check(session, engine):
        await user(session, "Check both inventories.")
        await call(session, "first")
        await call(session, "second")
        await output(session, "first", "FIRST_RESULT")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        assert not due(engine.calls[0])  # This response answers the initial user.
        engine.releases[0].set()
        await wait_for(lambda: session.current.done)
        await output(session, "second", "SECOND_RESULT")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2)
        assert [value["call_id"] for value in due(engine.calls[1])] == ["second"]
        assert sum("output" in value and value["call_id"] == "second"
                   for value in notices(engine.calls[1])) == 1
        engine.releases[1].set()
        await wait_for(lambda: session.current.done)
        await session.handle({"type": "response.create"})
        assert len(engine.calls) == 2

    asyncio.run(setup(check))


@pytest.mark.parametrize(("reason", "arrival"), [
    ("superseded", "before_cancel"), ("superseded", "after_cancel"), ("cancelled", "after_cancel")])
def test_obsolete_reserved_result_cannot_trigger_delivery_after_current_user_reply(reason, arrival):
    async def check(session, engine):
        await call(session)
        await user(session, "Answer this new question while the lookup runs.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        # Only superseding can invalidate an already completed task; ordinary
        # cancellation correctly leaves a completed result available.
        if arrival == "before_cancel":
            await output(session, value="OBSOLETE_RESULT")
            await session.handle({"type": "response.create"})
        await session.handle({"type": "frankie.task.cancel", "call_id": "lookup_1", "reason": reason})
        if arrival == "after_cancel":
            await output(session, value="OBSOLETE_RESULT")
            await session.handle({"type": "response.create"})
        engine.releases[0].set()
        await wait_for(lambda: session.current.done)
        await session.handle({"type": "response.create"})
        assert len(engine.calls) == 1 and not session.queued_task_response
        await user(session, "What happened to the lookup? Do not start it again.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2)
        assert "OBSOLETE_RESULT" not in str(engine.calls[1]) and not due(engine.calls[1])

    asyncio.run(setup(check))
