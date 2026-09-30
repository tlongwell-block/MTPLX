"""Terminal speculative replies need commitment even when no emit blocks them."""
import asyncio
import json
from threading import Event

import numpy as np
import pytest
from test_frankie_background_session import setup, wait_for
from test_frankie_regeneration import events

TOOL = '<tool_call>{"name":"lookup","arguments":{"topic":"weather"}}</tool_call>'
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"topic": {"type": "string"}}}}}]


async def immediate_speculation(s, engine, raw):
    returned = Event()

    def respond(history, settings, emit, abort, **kwargs):
        engine.calls.append(history)
        returned.set()
        # A tool-only result or immediate EOS has no public chunk to emit.
        return {"raw_text": raw, "finish_reason": "stop", "stats": {},
                "audio_seconds": 0, "seconds": .001}

    engine.respond = respond
    await s.handle({"type": "session.update", "session": {"tools": TOOLS}})
    run = s.start(s.audio_item(np.zeros(2400, dtype=np.float32)), tentative=True)
    await wait_for(returned.is_set)
    # The inference owner remains available while the event loop holds the result.
    assert await asyncio.wait_for(s.loop.run_in_executor(s.executor, lambda: True), .5)
    await asyncio.sleep(.02)
    return run


@pytest.mark.parametrize("raw", ["", TOOL], ids=["immediate-eos", "tool-only"])
@pytest.mark.parametrize("modalities", [["text"], ["audio"]], ids=["text", "audio"])
def test_unemitted_terminal_result_waits_for_commit_then_publishes_once(raw, modalities):
    async def check(s, e):
        await s.handle({"type": "session.update", "session": {"output_modalities": modalities}})
        run = await immediate_speculation(s, e, raw)
        assert not run.done and not run.visible and not run.ready.is_set()
        assert not s.task_ledger.tasks
        assert not any(event["type"].startswith("response.") for event in events(s))
        s.show(run)
        s.spec = None
        await wait_for(lambda: run.done)
        published = events(s)
        done = [event["response"] for event in published if event["type"] == "response.done"]
        assert len(done) == 1 and done[0]["id"] == run.id and done[0]["status"] == "completed"
        calls = [event for event in published if event["type"] == "response.function_call_arguments.done"]
        assert len(calls) == int(bool(raw))
        if raw:
            assert calls[0]["name"] == "lookup"
            assert json.loads(calls[0]["arguments"]) == {"topic": "weather"}
            assert s.task_ledger.tasks[calls[0]["call_id"]].status == "running"
        s.show(run)
        await asyncio.sleep(.02)
        assert events(s) == []
    asyncio.run(setup(check))


@pytest.mark.parametrize("action", ["discard", "cancel", "close"])
def test_aborted_invisible_tool_result_is_never_published(action):
    async def check(s, e):
        run = await immediate_speculation(s, e, TOOL)
        assert not run.done and not run.visible
        if action == "discard":
            s.discard_spec()
        elif action == "cancel":
            s.cancel()
        else:
            await asyncio.wait_for(s.close(), .5)
        await wait_for(lambda: run.done)
        assert run.abort.is_set() and not run.visible and not s.task_ledger.tasks
        assert not any(item["type"] == "function_call" for item in s.items)
        assert not any(event["type"].startswith("response.") for event in events(s))
    asyncio.run(setup(check))
