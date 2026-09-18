"""Background results cannot outrun playback or a user's request for silence."""

import asyncio
import time

import pytest
from test_frankie_background_session import call, output, setup, user, wait_for
from test_frankie_semantic_session import (
    Listener,
    committed_overlap,
    completed_run,
    enable,
    feed,
)


async def feedback(session, run, *, finished=False, position=None):
    await session.handle({
        "type": "frankie.playback.finished" if finished else "frankie.playback.position",
        "item_id": run.item_id, "response_id": run.id,
        "audio_end_ms": int(run.emitted_ms if position is None else position),
    })


def test_result_only_response_waits_for_actual_drain_even_after_wall_clock_deadline():
    async def check(session, engine):
        await call(session)
        run = completed_run(session)
        run.first_audio_at = time.monotonic() - 60
        await feedback(session, run, position=500)
        assert session.audible_response() is run
        await output(session)
        await session.handle({"type": "response.create"})
        assert session.queued_task_response and session.current is run
        assert not engine.calls
        await feedback(session, run, finished=True)
        await wait_for(lambda: len(engine.calls) == 1)
        assert session.current is not run
        assert not run.abort.is_set()
        assert "Heard. Never heard." in str(engine.calls[0])
        assert session.playback_wake is None

    asyncio.run(setup(check))


def test_clients_without_feedback_get_bounded_result_scheduling_fallback():
    async def check(session, engine):
        await call(session)
        run = completed_run(session)
        run.first_audio_at = time.monotonic() - run.emitted_ms / 1000 - 0.48
        await output(session)
        await session.handle({"type": "response.create"})
        assert not engine.calls and session.playback_wake is not None
        await wait_for(lambda: len(engine.calls) == 1)
        assert session.playback_wake is None

    asyncio.run(setup(check))


def test_typed_input_replaces_unheard_audio_before_new_history_snapshot():
    async def check(session, engine):
        run = completed_run(session)
        await feedback(session, run, position=500)
        await user(session, "Actually, change the day.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        assert run.abort.is_set()
        assert "Never heard." not in str(engine.calls[0])
        assert "Heard." in str(engine.calls[0])

    asyncio.run(setup(check))


def test_vad_interrupts_a_completed_but_still_audible_response():
    async def check(session, engine):
        enable(session, Listener())
        session.settings["interruption_policy"] = "vad"
        run = completed_run(session)
        await feedback(session, run, position=500)
        await feed(session, 16000, 3)
        assert run.abort.is_set()
        assert run.item["content"][0]["transcript"] == "Heard."
        assert session.metrics["barge_ins"] == 1

    asyncio.run(setup(check))


@pytest.mark.parametrize("source", ["semantic", "manual"])
def test_stop_stays_silent_when_background_result_arrives_until_new_input(source):
    async def check(session, engine):
        await call(session)
        enable(session, Listener("stop"))
        run = completed_run(session)
        await feedback(session, run, position=500)
        if source == "semantic":
            await session.schedule_overlap(committed_overlap(session), run)
        else:
            await session.handle({"type": "response.cancel", "response_id": run.id})
        assert run.abort.is_set() and session.speech_paused
        assert session.task_ledger.tasks["lookup_1"].status == "running"
        await output(session)
        await session.handle({"type": "response.create"})
        session.maybe_start_task_response()
        await asyncio.sleep(0.01)
        assert not engine.calls  # No spoken acknowledgment or unsolicited result.
        assert session.unhandled_task_results == {"lookup_1"}
        await user(session, "Okay, what did you find?")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        assert not session.speech_paused
        assert "Never heard." not in str(engine.calls[0])
        assert "Background task notice" in str(engine.calls[0])

    asyncio.run(setup(check))


def test_stop_does_not_generate_a_replacement_reply_without_background_mode():
    async def check(session, engine):
        session.settings["background_tasks"] = False
        enable(session, Listener("stop"))
        run = completed_run(session)
        await session.schedule_overlap(committed_overlap(session), run)
        await session.handle({"type": "response.create"})
        assert session.speech_paused and not engine.calls

    asyncio.run(setup(check))


def test_superseded_listener_cancellation_does_not_surface_as_a_user_error():
    async def check(session, engine):
        listener = Listener("adapt", RuntimeError("Cancelled observation"), held=True)
        enable(session, listener)
        run = completed_run(session)
        task = session.schedule_overlap(committed_overlap(session), run)
        await listener.entered.wait()
        session.invalidate_semantics()
        listener.release.set()
        await task
        events = []
        while not session.outgoing.empty():
            events.append(session.outgoing.get_nowait())
        assert not any(event["type"] == "frankie.interaction" for event in events)
        assert not engine.calls and not session.semantic_pending

    asyncio.run(setup(check))
