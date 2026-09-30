"""Tool completion is separate from a successful, played presentation attempt."""

import asyncio
import copy

import numpy as np
import pytest
from test_frankie_background_session import call, output, setup, user, wait_for
from test_frankie_playback_tasks import feedback
from test_frankie_task_delivery import due


async def initial_call(session, engine):
    await user(session, "Check the inventory.")
    await call(session)
    await session.handle({"type": "response.create"})
    await wait_for(lambda: len(engine.calls) == 1)
    engine.releases[0].set()
    await wait_for(lambda: session.current.done)


def install_reply(engine, outcome="speech", *, held=False):
    def respond(history, settings, emit, abort, **kwargs):
        index = len(engine.calls)
        engine.calls.append(copy.deepcopy(history))
        text = "Thirteen units are available, dispatching Thursday."
        if outcome in {"speech", "length", "text_no_audio"}:
            emit("text", text)
            if "audio" in settings["output_modalities"] and outcome != "text_no_audio":
                emit("audio", np.zeros(24000, dtype=np.float32))
                emit("chunk", {"text": text, "end_ms": 1000})
        if held:
            while not engine.releases[index].wait(0.002) and not abort.is_set():
                pass
        if outcome == "failed":
            raise RuntimeError("Injected backend failure")
        return {"raw_text": text if outcome in {"speech", "length"} else "",
                "finish_reason": "length" if outcome == "length" else "stop",
                "stats": {}, "audio_seconds": 0, "seconds": 0.001}
    engine.respond = respond


@pytest.mark.parametrize("outcome", ["empty", "failed", "length"])
def test_unsuccessful_result_stays_available_without_automatic_retry(outcome):
    async def check(session, engine):
        await initial_call(session, engine)
        install_reply(engine, outcome)
        await output(session, value="THIRTEEN_THURSDAY")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2 and session.current.done)
        task = session.task_ledger.tasks["lookup_1"]
        assert task.status == "completed" and not task.consumed
        assert task.delivery_response_id is None and task.delivery_failed
        assert session.unhandled_task_results == {"lookup_1"}
        assert not session.ready_task_results() and not session.queued_task_response
        for _ in range(3):
            session.maybe_start_task_response()
            await asyncio.sleep(0)
        assert len(engine.calls) == 2
        # One explicit idle request can retry; the failed attempt itself cannot.
        install_reply(engine)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 3 and session.current.done)
        assert task.consumed and not task.delivery_failed and task.delivery_response_id is None
        assert not session.unhandled_task_results
        assert [value["call_id"] for value in due(engine.calls[-1])] == ["lookup_1"]
        await session.handle({"type": "response.create"})
        assert len(engine.calls) == 3

    asyncio.run(setup(check))


def test_repeated_failed_explicit_attempt_still_does_not_self_retry():
    async def check(session, engine):
        await initial_call(session, engine)
        install_reply(engine, "empty", held=True)
        await output(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2)
        for _ in range(3):
            await session.handle({"type": "response.create"})
        engine.releases[1].set()
        await wait_for(lambda: session.current.done)
        assert len(engine.calls) == 2
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 3)
        engine.releases[2].set()
        await wait_for(lambda: session.current.done)
        session.maybe_start_task_response()
        assert len(engine.calls) == 3
        assert session.task_ledger.tasks["lookup_1"].delivery_failed

    asyncio.run(setup(check))


def test_transcript_without_requested_audio_is_not_delivery():
    async def check(session, engine):
        await initial_call(session, engine)
        session.settings.update(output_modalities=["audio"], playback_feedback=True)
        install_reply(engine, "text_no_audio")
        await output(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2 and session.current.done)
        task = session.task_ledger.tasks["lookup_1"]
        assert session.current.text and session.current.emitted_ms == 0
        assert task.delivery_failed and not task.consumed and session.unhandled_task_results == {"lookup_1"}

    asyncio.run(setup(check))


def test_listener_accepted_input_alone_does_not_unblock_failed_delivery():
    async def check(session, engine):
        await initial_call(session, engine)
        install_reply(engine, "empty")
        await output(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2 and session.current.done)
        # A nod can enter the authoritative input history and then be handled
        # without any new generation. That must not trigger an old failed task.
        session.accept_input(semantic_overlap=True)
        session.response_revision = session.user_revision
        session.queued_task_response = True
        session.maybe_start_task_response()
        await asyncio.sleep(0)
        assert len(engine.calls) == 2 and not session.ready_task_results()

    asyncio.run(setup(check))


@pytest.mark.parametrize("with_feedback", [False, True])
def test_nonempty_audio_delivery_finishes_only_at_required_playback_frontier(with_feedback):
    async def check(session, engine):
        await initial_call(session, engine)
        session.settings.update(output_modalities=["audio"], playback_feedback=with_feedback)
        install_reply(engine)
        await output(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2 and session.current.done)
        run, task = session.current, session.task_ledger.tasks["lookup_1"]
        assert not session.unhandled_task_results
        if with_feedback:
            assert not task.consumed and task.delivery_response_id == run.id
            await feedback(session, run, position=1000)
            assert not task.consumed  # Position alone is not actual drain/completion.
            await feedback(session, run, finished=True)
            await feedback(session, run, finished=True)
        assert task.consumed and task.delivery_response_id is None
        assert not session.unhandled_task_results
        assert len(engine.calls) == 2

    asyncio.run(setup(check))


@pytest.mark.parametrize("generation_done", [False, True])
def test_unheard_cancelled_delivery_stays_silent_until_new_input_then_can_retry(generation_done):
    async def check(session, engine):
        await initial_call(session, engine)
        session.settings.update(output_modalities=["audio"], playback_feedback=True)
        install_reply(engine, held=not generation_done)
        await output(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: session.current.emitted_ms == 1000 and bool(session.current.chunks))
        old, task = session.current, session.task_ledger.tasks["lookup_1"]
        if generation_done:
            await wait_for(lambda: old.done)
        await session.handle({"type": "response.cancel", "response_id": old.id})
        await wait_for(lambda: old.done)
        assert not task.consumed and task.delivery_failed
        assert session.unhandled_task_results == {"lookup_1"}
        assert old.played_ms == 0 and not old.item["content"][0]["transcript"]
        await session.handle({"type": "response.create"})
        session.maybe_start_task_response()
        assert len(engine.calls) == 2 and session.speech_paused
        install_reply(engine)
        await user(session, "Okay, what did the lookup find?")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 3 and session.current.done)
        replacement = session.current
        assert not due(engine.calls[-1])  # Actual user request retains priority.
        assert task.delivery_response_id == replacement.id and not task.consumed
        await feedback(session, old, finished=True)  # Cannot settle the new reservation.
        assert task.delivery_response_id == replacement.id and not task.consumed
        await feedback(session, replacement, finished=True)
        assert task.consumed and not session.unhandled_task_results

    asyncio.run(setup(check))


def test_superseded_reserved_result_clears_obsolete_playback_and_never_revives():
    async def check(session, engine):
        await initial_call(session, engine)
        session.settings.update(output_modalities=["audio"], playback_feedback=True)
        install_reply(engine)
        await output(session, value="OBSOLETE_RESULT")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2 and session.current.done)
        run, task = session.current, session.task_ledger.tasks["lookup_1"]
        await session.handle({"type": "frankie.task.cancel", "call_id": "lookup_1", "reason": "superseded"})
        assert run.abort.is_set() and task.status == "superseded" and not task.consumed
        assert task.delivery_response_id is None and not session.unhandled_task_results
        await feedback(session, run, finished=True)
        assert not task.consumed
        await user(session, "Explain what happened without starting another lookup.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 3)
        assert "OBSOLETE_RESULT" not in str(engine.calls[-1]) and not due(engine.calls[-1])

    asyncio.run(setup(check))


def test_new_result_can_follow_failed_attempt_without_retrying_its_old_result():
    async def check(session, engine):
        await initial_call(session, engine)
        await call(session, "second")
        install_reply(engine, "failed", held=True)
        await output(session, value="FIRST_RESULT")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 2)
        await output(session, "second", "SECOND_RESULT")
        await session.handle({"type": "response.create"})
        install_reply(engine)  # Future owner job only; current closure still fails.
        engine.releases[1].set()
        await wait_for(lambda: len(engine.calls) == 3 and session.current.done)
        assert [value["call_id"] for value in due(engine.calls[-1])] == ["second"]
        assert session.task_ledger.tasks["lookup_1"].delivery_failed
        assert not session.task_ledger.tasks["lookup_1"].consumed
        assert session.task_ledger.tasks["second"].consumed
        assert session.unhandled_task_results == {"lookup_1"}
        assert not session.queued_task_response

    asyncio.run(setup(check))
