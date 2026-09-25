"""Ordinary realtime tool results must not interrupt audible speech."""
import asyncio
import time

import pytest

from test_frankie_background_session import call, output, setup, user, wait_for
from mtplx.frankie.session import Response


async def audible_call(session, feedback):
    session.settings.update(background_tasks=False, playback_feedback=feedback)
    await call(session)
    item = {"id": "speaking", "type": "message", "role": "assistant",
            "content": [{"type": "output_audio", "transcript": "First. Last."}]}
    run = Response(visible=True, done=True, status="completed", settings=dict(session.settings),
                   item=item, text="First. Last.", emitted_ms=2000, played_ms=1000,
                   first_audio_at=time.monotonic())
    run.item_id = item["id"]
    run.chunks = [{"text": "First.", "end_ms": 1000}, {"text": "Last.", "end_ms": 2000}]
    session.items.insert(0, item)
    session.current = run
    session.playback_runs[run.item_id] = run
    session.user_revision = session.response_revision = 1
    await output(session, value="lighthouse")
    return run


@pytest.mark.parametrize("feedback", [False, True])
def test_standard_result_waits_for_drain_and_coalesces_requests(feedback):
    async def check(session, engine):
        run = await audible_call(session, feedback)
        for _ in range(2):
            await session.handle({"type": "response.create"})
        assert not run.abort.is_set() and session.current is run
        assert run.item["content"][0]["transcript"] == "First. Last."
        assert not engine.calls and session.queued_task_response
        if feedback:
            await session.handle({"type": "frankie.playback.finished", "item_id": run.item_id,
                                  "response_id": run.id, "audio_end_ms": 2000})
        else:
            run.first_audio_at -= 10
            session.playback_deadline()
        await wait_for(lambda: len(engine.calls) == 1)
        assert not run.abort.is_set()
        assert run.item["content"][0]["transcript"] == "First. Last."
        assert [i["output"] for i in engine.calls[0] if i["type"] == "function_call_output"] == ["lighthouse"]
        engine.releases[0].set()
        await wait_for(lambda: session.current.done)
        session.playback_deadline()
        assert len(engine.calls) == 1 and not session.queued_task_response

    asyncio.run(setup(check))


def test_new_user_turn_can_interrupt_speech_with_queued_standard_result():
    async def check(session, engine):
        run = await audible_call(session, True)
        await session.handle({"type": "response.create"})
        await user(session, "A new question.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        assert run.abort.is_set() and run.item["content"][0]["transcript"] == "First."
        assert not session.queued_task_response
        assert any(i.get("role") == "user" and i["content"][0]["text"] == "A new question."
                   for i in engine.calls[0])

    asyncio.run(setup(check))


def test_next_turn_without_playback_feedback_preserves_completed_reply():
    async def check(session, engine):
        run = await audible_call(session, False)
        # An ordinary WebSocket client supplies neither playback positions nor
        # a truncate event. A conservative drain estimate is not an interruption.
        run.played_ms = 0
        await user(session, "A new question.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(engine.calls) == 1)
        assert not run.abort.is_set() and not run.interrupted
        assert run.item["content"][0]["transcript"] == "First. Last."
        previous = next(i for i in engine.calls[0] if i["id"] == run.item_id)
        assert previous["content"][0]["transcript"] == "First. Last."
        assert not any(e["type"] == "frankie.playback.clear"
                       for e in list(session.outgoing._queue))

    asyncio.run(setup(check))


def test_cancel_prevents_queued_standard_result_from_starting_after_drain():
    async def check(session, engine):
        run = await audible_call(session, True)
        await session.handle({"type": "response.create"})
        await session.handle({"type": "response.cancel"})
        await session.handle({"type": "frankie.playback.finished", "item_id": run.item_id,
                              "response_id": run.id, "audio_end_ms": 2000})
        await asyncio.sleep(0)
        assert not engine.calls and not session.queued_task_response

    asyncio.run(setup(check))


def test_capability_changes_cannot_strand_a_queued_standard_result():
    async def check(session, engine):
        run = await audible_call(session, True)
        await session.handle({"type": "response.create"})
        with pytest.raises(ValueError, match="queued task"):
            await session.handle({"type": "session.update", "session": {
                "frankie": {"background_tasks": True}}})
        assert not session.settings["background_tasks"] and session.queued_task_response
        run.first_audio_at -= 10
        await session.handle({"type": "session.update", "session": {
            "frankie": {"playback_feedback": False}}})
        await wait_for(lambda: len(engine.calls) == 1)
        assert not run.abort.is_set()

    asyncio.run(setup(check))


def test_pending_user_audio_has_priority_over_standard_result_followup():
    async def check(session, engine):
        run = await audible_call(session, True)
        session.settings["streaming_listener"] = "backchannel"
        session.listening = True
        await session.handle({"type": "response.create"})
        await session.handle({"type": "frankie.playback.finished", "item_id": run.item_id,
                              "response_id": run.id, "audio_end_ms": 2000})
        assert not engine.calls and session.queued_task_response
        await session.handle({"type": "input_audio_buffer.clear"})
        await wait_for(lambda: len(engine.calls) == 1)
        assert not run.abort.is_set()

    asyncio.run(setup(check))
