"""A slow ordinary tool may finish while another reply is generating."""
import asyncio
import copy

import pytest

from test_frankie_background_session import call, setup, user, wait_for


def result(call_id="lookup_1", value="lighthouse", item_id="late_result"):
    return {"type": "conversation.item.create", "event_id": "result_event",
            "item": {"id": item_id, "type": "function_call_output",
                     "call_id": call_id, "output": value}}


def test_ordinary_result_acknowledged_during_generation_and_used_next_turn():
    async def check(session, engine):
        session.settings["background_tasks"] = False
        await call(session)
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        active = session.current
        original = copy.deepcopy(engine.calls[0])
        await session.handle(result())
        acks = [e for e in session.outgoing._queue
                if e["type"] == "conversation.item.created"
                and e.get("item", {}).get("id") == "late_result"]
        assert len(acks) == 1 and acks[0]["item"]["output"] == "lighthouse"
        assert session.current is active and not active.abort.is_set()
        assert not active.done and engine.calls[0] == original
        assert not any(i["type"] == "function_call_output" for i in original)
        assert "lookup_1" in session.unhandled_task_results
        with pytest.raises(ValueError, match="already supplied"):
            await session.handle(result(item_id="duplicate_result"))
        with pytest.raises(ValueError, match="Cancel the active response"):
            await user(session, "Unrelated history edits remain guarded.")
        engine.releases[0].set()
        await wait_for(lambda: active.done)
        assert len(engine.calls) == 1  # The client owns continuation.
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2)
        assert [i["output"] for i in engine.calls[1]
                if i["type"] == "function_call_output"] == ["lighthouse"]
        engine.releases[1].set()
        await wait_for(lambda: session.current.done)
        assert not session.unhandled_task_results
        assert not active.abort.is_set()

    asyncio.run(setup(check))


@pytest.mark.parametrize("event, message", [
    (result(call_id="unknown"), "Unknown tool call"),
    (result(value={"bad": "not a string"}), "Tool output must be a string"),
])
def test_invalid_active_tool_results_do_not_change_history(event, message):
    async def check(session, engine):
        session.settings["background_tasks"] = False
        await call(session)
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        active = session.current
        original = copy.deepcopy(session.items)
        with pytest.raises(ValueError, match=message):
            await session.handle(event)
        assert session.items == original and not session.unhandled_task_results
        assert session.current is active and not active.abort.is_set()
        assert not any(e["type"] == "conversation.item.created"
                       and e.get("item", {}).get("id") == "late_result"
                       for e in session.outgoing._queue)

    asyncio.run(setup(check))
