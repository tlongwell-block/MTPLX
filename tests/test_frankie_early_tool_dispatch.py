"""Complete committed calls can run while speech drains, never speculatively."""

import asyncio
import copy

import numpy as np
import pytest
from test_frankie_background_session import output, setup, user, wait_for
from test_frankie_playback_tasks import feedback

RAW = '<tool_call>{"name":"lookup","arguments":{"topic":"weather"}}</tool_call>'


def candidate(position=30, call_id="issued-1"):
    return {"key": (position, 0), "call": {"id": call_id, "type": "function",
            "function": {"name": "lookup", "arguments": '{"topic":"weather"}'}}}


def seen(session, kind):
    return [event for event in session.outgoing._queue if event["type"] == kind]


def install(engine, *, candidates=None, held=True, failure=False, speech=True):
    candidates = [candidate()] if candidates is None else candidates

    def respond(history, settings, emit, abort, **kwargs):
        index = len(engine.calls)
        engine.calls.append(copy.deepcopy(history))
        if index == 0:
            if speech:
                emit("text", "I can check that.")
                if "audio" in settings["output_modalities"]:
                    emit("audio", np.zeros(24000, dtype=np.float32))
            for value in candidates:
                emit("tool_call", value)
            if held:
                while not engine.releases[index].wait(.002) and not abort.is_set():
                    pass
            if failure:
                raise RuntimeError("Injected failure after tool closure")
        else:
            emit("text", "The result is forty two.")
            if "audio" in settings["output_modalities"]:
                emit("audio", np.zeros(24000, dtype=np.float32))
        return {"raw_text": RAW if index == 0 else "The result is forty two.",
                "tool_events": candidates if index == 0 else [], "finish_reason": "stop",
                "stats": {}, "audio_seconds": 1 if speech else 0, "seconds": .01}

    engine.respond = respond


def test_call_is_published_during_generation_once_with_stable_final_identity():
    async def check(session, engine):
        value = candidate()
        install(engine, candidates=[value, copy.deepcopy(value)])
        await user(session, "Check the weather.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: bool(seen(session, "response.function_call_arguments.done")))
        run = session.current
        assert not run.done and len(engine.calls) == 1
        assert len(seen(session, "response.function_call_arguments.done")) == 1
        assert seen(session, "response.done") == []
        assert session.task_ledger.tasks["issued-1"].status == "running"
        engine.releases[0].set()
        await wait_for(lambda: run.done)
        done = seen(session, "response.done")[-1]["response"]
        calls = [item for item in done["output"] if item["type"] == "function_call"]
        assert len(calls) == 1 and calls[0]["call_id"] == "issued-1"
        assert seen(session, "response.function_call_arguments.done")[0]["output_index"] == 1
        # Even a new late candidate cannot appear after response.done.
        session.publish(run, "tool_call", candidate(50, "late"))
        assert len(seen(session, "response.function_call_arguments.done")) == 1

    asyncio.run(setup(check))


@pytest.mark.parametrize("action", ["commit", "discard", "close", "error"])
def test_tentative_candidates_free_owner_and_have_no_effect_before_commit(action):
    async def check(session, engine):
        install(engine, held=False, speech=False, failure=action == "error")
        run = session.start(session.audio_item(np.zeros(2400, dtype=np.float32)), tentative=True)
        await wait_for(lambda: bool(run.tool_candidates))
        assert await asyncio.wait_for(session.loop.run_in_executor(session.executor, lambda: True), .5)
        if action == "error":
            await wait_for(lambda: run.generation_failed)
        assert not run.visible and not session.task_ledger.tasks
        assert not [item for item in session.items if item["type"] == "function_call"]
        assert seen(session, "response.function_call_arguments.done") == []
        if action in {"commit", "error"}:
            session.show(run)
            session.spec = None
        elif action == "discard":
            session.discard_spec()
        else:
            await session.close()
        await wait_for(lambda: run.done)
        assert len(seen(session, "response.function_call_arguments.done")) == (action == "commit")
        if action == "error":
            assert seen(session, "response.done")[-1]["response"]["status"] == "failed"

    asyncio.run(setup(check))


@pytest.mark.parametrize("outcome", ["cancel", "error"])
def test_issued_work_survives_later_response_cancel_or_failure(outcome):
    async def check(session, engine):
        install(engine, failure=outcome == "error")
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: bool(session.current.tool_items))
        run = session.current
        if outcome == "cancel":
            session.cancel()
        engine.releases[0].set()
        await wait_for(lambda: run.done)
        response = seen(session, "response.done")[-1]["response"]
        assert response["status"] == ("cancelled" if outcome == "cancel" else "failed")
        assert [item["call_id"] for item in response["output"]
                if item["type"] == "function_call"] == ["issued-1"]
        assert session.task_ledger.tasks["issued-1"].status == "running"
        # Stopping work is a separate explicit action; late results are discarded.
        await session.handle({"type": "frankie.task.cancel", "call_id": "issued-1"})
        await output(session, "issued-1")
        assert session.task_ledger.tasks["issued-1"].status == "cancelled"
        assert not session.unhandled_task_results

    asyncio.run(setup(check))


def test_result_during_acknowledgment_queues_one_followup_after_actual_drain():
    async def check(session, engine):
        session.settings.update(output_modalities=["audio"], playback_feedback=True)
        install(engine)
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: bool(session.current.tool_items))
        run = session.current
        frozen = copy.deepcopy(engine.calls[0])
        await output(session, "issued-1", "FORTY_TWO")
        for _ in range(2):
            await session.handle({"type": "response.create"})
        assert len(engine.calls) == 1 and not run.abort.is_set()
        assert engine.calls[0] == frozen and session.queued_task_response
        engine.releases[0].set()
        await wait_for(lambda: run.done)
        await asyncio.sleep(.02)
        assert len(engine.calls) == 1
        await feedback(session, run, finished=True)
        await wait_for(lambda: len(engine.calls) == 2 and session.current.done)
        assert "FORTY_TWO" in str(engine.calls[1])
        await feedback(session, session.current, finished=True)
        await session.handle({"type": "response.create"})
        assert len(engine.calls) == 2 and session.task_ledger.tasks["issued-1"].consumed

    asyncio.run(setup(check))


def test_distinct_identical_calls_are_not_collapsed_and_limit_failure_preserves_issued_call():
    async def check(session, engine):
        session.task_ledger.max_pending = 1
        install(engine, candidates=[candidate(), candidate(60, "issued-2")], speech=False)
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: session.current.done)
        response = seen(session, "response.done")[-1]["response"]
        assert response["status"] == "failed"
        assert [item["call_id"] for item in response["output"]
                if item["type"] == "function_call"] == ["issued-1"]
        assert len(seen(session, "response.function_call_arguments.done")) == 1

    asyncio.run(setup(check))


def test_two_identical_calls_retain_two_distinct_stable_ids_and_output_indices():
    async def check(session, engine):
        install(engine, candidates=[candidate(), candidate(60, "issued-2")], held=False)
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: session.current.done)
        calls = seen(session, "response.function_call_arguments.done")
        assert [call["call_id"] for call in calls] == ["issued-1", "issued-2"]
        assert [call["output_index"] for call in calls] == [1, 2]

    asyncio.run(setup(check))


def test_legacy_capability_buffers_candidates_until_terminal():
    async def check(session, engine):
        await session.handle({"type": "session.update", "session": {
            "frankie": {"background_tasks": False}}})
        install(engine)
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: bool(session.current.tool_candidates))
        assert not session.task_ledger.tasks
        assert seen(session, "response.function_call_arguments.done") == []
        engine.releases[0].set()
        await wait_for(lambda: session.current.done)
        assert len(seen(session, "response.function_call_arguments.done")) == 1
        assert not session.task_ledger.tasks
        await output(session, "issued-1")  # Ordinary clients can now return it.

    asyncio.run(setup(check))


def test_disabling_background_during_generation_is_rejected_before_tool_closure():
    async def check(session, engine):
        value = candidate()

        def respond(history, settings, emit, abort, **kwargs):
            engine.calls.append(history)
            while not engine.releases[0].wait(.002) and not abort.is_set():
                pass
            emit("tool_call", value)
            while not engine.releases[1].wait(.002) and not abort.is_set():
                pass
            return {"raw_text": RAW, "tool_events": [value], "finish_reason": "stop",
                    "stats": {}, "audio_seconds": 0, "seconds": .01}

        engine.respond = respond
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: bool(engine.calls))
        with pytest.raises(ValueError, match="Wait for the current response"):
            await session.handle({"type": "session.update", "session": {
                "frankie": {"background_tasks": False}}})
        assert session.settings["background_tasks"]
        engine.releases[0].set()
        await wait_for(lambda: bool(session.current.tool_items))
        assert session.task_ledger.tasks["issued-1"].status == "running"
        assert len(seen(session, "response.function_call_arguments.done")) == 1
        assert not session.current.done
        engine.releases[1].set()
        await wait_for(lambda: session.current.done)
        assert len(seen(session, "response.function_call_arguments.done")) == 1
        await output(session, "issued-1")

    asyncio.run(setup(check))


def test_changed_candidate_cannot_replace_an_already_issued_call():
    async def check(session, engine):
        install(engine)
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: bool(session.current.tool_items))
        changed = candidate()
        changed["call"]["function"]["arguments"] = '{"topic":"changed"}'
        session.publish(session.current, "tool_call", changed)
        await wait_for(lambda: session.current.done)
        response = seen(session, "response.done")[-1]["response"]
        assert response["status"] == "failed"
        calls = [item for item in response["output"] if item["type"] == "function_call"]
        assert len(calls) == 1 and calls[0]["arguments"] == '{"topic":"weather"}'
        assert len(seen(session, "response.function_call_arguments.done")) == 1

    asyncio.run(setup(check))
