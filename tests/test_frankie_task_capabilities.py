"""Capability updates cannot strand an outstanding task presentation."""

import asyncio
import copy

import pytest
from test_frankie_background_session import call, output, setup, wait_for
from test_frankie_playback_tasks import feedback
from test_frankie_task_lifecycle import initial_call, install_reply


async def reserved_reply(session, engine):
    await initial_call(session, engine)
    session.settings.update(output_modalities=["audio"], playback_feedback=True)
    install_reply(engine)
    await output(session)
    await session.handle({"type": "response.create"})
    await wait_for(lambda: len(engine.calls) == 2 and session.current.done)
    task = session.task_ledger.tasks["lookup_1"]
    assert task.delivery_response_id == session.current.id and not task.consumed
    return session.current, task


@pytest.mark.parametrize("extension", ["playback_feedback", "background_tasks"])
def test_capability_disable_during_reserved_playback_is_atomic(extension):
    async def check(session, engine):
        run, task = await reserved_reply(session, engine)
        settings = copy.deepcopy(session.settings)
        ledger = session.task_ledger
        task_before = copy.deepcopy(vars(task))
        with pytest.raises(ValueError, match="task"):
            await session.handle({"type": "session.update", "session": {
                "frankie": {extension: False}, "temperature": 0.9,
            }})
        assert session.settings == settings and session.task_ledger is ledger
        assert vars(task) == task_before and session.current is run
        assert not session.unhandled_task_results
        await feedback(session, run, finished=True)
        assert task.consumed and task.delivery_response_id is None
        await session.handle({"type": "session.update", "session": {
            "frankie": {extension: False},
        }})
        assert not session.settings[extension]

    asyncio.run(setup(check))


def test_background_disable_rejects_failed_undelivered_result_until_superseded():
    async def check(session, engine):
        await initial_call(session, engine)
        install_reply(engine, "empty")
        await output(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2 and session.current.done)
        task = session.task_ledger.tasks["lookup_1"]
        assert task.delivery_failed and task.delivery_response_id is None
        with pytest.raises(ValueError, match="supersede"):
            await session.handle({"type": "session.update", "session": {
                "frankie": {"background_tasks": False},
            }})
        assert session.settings["background_tasks"] and session.unhandled_task_results == {"lookup_1"}
        await session.handle({"type": "frankie.task.cancel", "call_id": "lookup_1", "reason": "superseded"})
        await session.handle({"type": "session.update", "session": {
            "frankie": {"background_tasks": False},
        }})
        assert not session.settings["background_tasks"]

    asyncio.run(setup(check))


def test_unchanged_background_capability_is_allowed_while_delivery_drains():
    async def check(session, engine):
        run, task = await reserved_reply(session, engine)
        await session.handle({"type": "session.update", "session": {
            "frankie": {"background_tasks": True, "playback_feedback": True},
        }})
        assert task.delivery_response_id == run.id and not task.consumed

    asyncio.run(setup(check))


async def background(session, enabled):
    await session.handle({"type": "session.update", "session": {
        "frankie": {"background_tasks": enabled},
    }})


def test_reenable_preserves_consumed_task_identity_and_allows_later_disable():
    async def check(session, engine):
        run, task = await reserved_reply(session, engine)
        await feedback(session, run, finished=True)
        original = copy.deepcopy(vars(task))
        for _ in range(2):
            await background(session, False)
            await background(session, True)
            assert session.task_ledger.tasks["lookup_1"] is task
            assert vars(task) == original
            assert not session.unhandled_task_results
        await background(session, False)

    asyncio.run(setup(check))


def test_legacy_completed_results_are_historical_but_pending_calls_stay_live():
    async def check(session, engine):
        await background(session, False)
        await call(session, "historical")
        await output(session, "historical")
        await call(session, "pending")
        await background(session, True)
        ledger = session.task_ledger
        assert list(ledger.tasks) == ["historical", "pending"]
        assert ledger.tasks["historical"].consumed
        assert ledger.tasks["pending"].status == "running"
        assert not session.unhandled_task_results and not session.queued_task_response
        await output(session, "pending")
        assert session.unhandled_task_results == {"pending"}
        assert not ledger.tasks["pending"].consumed

    asyncio.run(setup(check))


def test_legacy_late_cancelled_output_stays_discarded_on_reenable():
    async def check(session, engine):
        await call(session)
        task = session.task_ledger.tasks["lookup_1"]
        await session.handle({"type": "frankie.task.cancel", "call_id": "lookup_1"})
        await background(session, False)
        await output(session)
        await background(session, True)
        assert session.task_ledger.tasks["lookup_1"] is task
        assert task.status == "cancelled" and task.received
        result = next(item for item in session.items if item["type"] == "function_call_output")
        assert result["_task_discarded"] and not session.unhandled_task_results

    asyncio.run(setup(check))


def test_reenable_admission_failure_is_atomic_and_does_not_advance_cursor():
    async def check(session, engine):
        await background(session, False)
        session.task_ledger.max_pending = 1
        await call(session, "first")
        await call(session, "second")
        ledger, cursor = session.task_ledger, session.task_history_count
        before = copy.deepcopy(session.items)
        with pytest.raises(ValueError, match="pending"):
            await background(session, True)
        assert session.task_ledger is ledger and not ledger.tasks
        assert session.task_history_count == cursor and session.items == before
        assert not session.settings["background_tasks"]

    asyncio.run(setup(check))


def test_reenable_keeps_capacity_order_without_resurrecting_evicted_history():
    async def check(session, engine):
        await background(session, False)
        session.task_ledger.max_entries = session.task_ledger.max_pending = 2
        for name in ("old", "middle", "new"):
            await call(session, name)
            await output(session, name)
        await background(session, True)
        assert list(session.task_ledger.tasks) == ["middle", "new"]
        tasks = list(session.task_ledger.tasks.values())
        await background(session, False)
        await background(session, True)
        assert list(session.task_ledger.tasks) == ["middle", "new"]
        assert all(current is previous for current, previous in zip(session.task_ledger.tasks.values(), tasks))
        assert not session.unhandled_task_results

    asyncio.run(setup(check))
