"""Long overlap yields the floor without truncating the user's utterance."""

import asyncio
import time

import numpy as np
import pytest
from test_frankie_background_session import call, output, setup, wait_for
from test_frankie_semantic_session import (
    Listener,
    committed_overlap,
    completed_run,
    enable,
    feed,
)


def speaking_run(session, listener):
    enable(session, listener)
    run = completed_run(session)
    run.done = False
    run.emitted_ms = 10000
    return run


def user_audio(history):
    return [part["_pcm"] for item in history if item.get("role") == "user"
            for part in item.get("content", []) if part["type"] == "input_audio"]


@pytest.mark.parametrize("padding_frames", [0, 10])
def test_cap_boundary_preserves_complete_audio_and_future_turns(padding_frames):
    async def check(session, engine):
        listener = Listener()
        run = speaking_run(session, listener)
        await feed(session, 0, padding_frames)
        before_cap = 187 - padding_frames  # 187 * 32 ms = 5.984 seconds.
        await feed(session, 16000, before_cap)
        assert not run.abort.is_set() and session.overlap_run is run
        assert sum(map(len, session.frames)) == 187 * 768
        await feed(session, 16000, 1)  # 6.016 seconds, including initial padding.
        assert run.abort.is_set() and session.overlap_run is None
        assert session.listening and session.metrics["barge_ins"] == 1
        await feed(session, 16000, 15)
        assert not engine.calls and not listener.entered.is_set()
        assert session.metrics["barge_ins"] == 1  # No repeated cancel on later frames.
        await feed(session, 0, 10)
        await wait_for(lambda: len(engine.calls) == 1)
        captured = user_audio(engine.calls[0])[-1]
        assert np.count_nonzero(captured) == (before_cap + 16) * 768
        assert len(captured) > 6 * 24000
        assert np.all(captured[:padding_frames * 768] == 0)
        assert not session.listening and not session.semantic_pending
        engine.releases[0].set()
        await wait_for(lambda: session.current.done)
        await feed(session, 16000, 4)
        await feed(session, 0, 10)
        await wait_for(lambda: len(engine.calls) == 2)
        assert np.count_nonzero(user_audio(engine.calls[1])[-1]) == 4 * 768
        assert session.metrics["barge_ins"] == 1

    asyncio.run(setup(check))


def test_endpoint_padding_crossing_cap_uses_normal_response_instead_of_rejected_probe():
    async def check(session, engine):
        listener = Listener()
        run = speaking_run(session, listener)
        await feed(session, 16000, 186)
        assert not run.abort.is_set()
        await feed(session, 0, 1)  # 5.984 seconds including endpoint padding.
        assert session.overlap_run is run
        await feed(session, 0, 1)  # The second silence frame crosses six seconds.
        assert run.abort.is_set() and session.overlap_run is None
        await feed(session, 0, 8)
        await wait_for(lambda: len(engine.calls) == 1)
        assert np.count_nonzero(user_audio(engine.calls[0])[-1]) == 186 * 768
        assert not listener.entered.is_set()

    asyncio.run(setup(check))


def test_pending_tool_result_does_not_start_speaking_during_long_user_turn():
    async def check(session, engine):
        session.settings["playback_feedback"] = True
        await call(session)
        run = speaking_run(session, Listener())
        await feed(session, 16000, 180)
        await output(session, value="LONG_TURN_RESULT_42")
        await session.handle({"type": "response.create"})
        assert session.queued_task_response
        await feed(session, 16000, 20)
        assert run.abort.is_set() and session.listening
        assert session.unhandled_task_results == {"lookup_1"}
        session.maybe_start_task_response()
        assert not engine.calls
        await feed(session, 0, 10)
        await wait_for(lambda: len(engine.calls) == 1)
        assert np.count_nonzero(user_audio(engine.calls[0])[-1]) == 200 * 768
        assert "Background task notice" not in str(engine.calls[0])
        assert "LONG_TURN_RESULT_42" not in str(engine.calls[0])
        assert session.queued_task_response
        assert session.unhandled_task_results == {"lookup_1"}
        assert not session.task_ledger.tasks["lookup_1"].consumed
        replacement = session.current
        replacement.emitted_ms = 1000
        replacement.first_audio_at = time.monotonic()
        engine.releases[0].set()
        await wait_for(lambda: replacement.done)
        assert len(engine.calls) == 1  # Generation end is not playback drain.
        await session.handle({
            "type": "frankie.playback.finished", "item_id": replacement.item_id,
            "response_id": replacement.id, "audio_end_ms": 1000,
        })
        await wait_for(lambda: len(engine.calls) == 2)
        assert "LONG_TURN_RESULT_42" in str(engine.calls[1])
        assert not session.queued_task_response and not session.unhandled_task_results
        task = session.task_ledger.tasks["lookup_1"]
        assert not task.consumed and task.delivery_response_id == session.current.id
        engine.releases[1].set()
        await wait_for(lambda: session.current.done)
        assert task.consumed and task.delivery_response_id is None
        await session.handle({"type": "response.create"})
        session.maybe_start_task_response()
        await asyncio.sleep(0.01)
        assert len(engine.calls) == 2

    asyncio.run(setup(check))


def test_already_drained_reply_is_not_falsely_interrupted_when_cap_is_crossed():
    async def check(session, engine):
        run = speaking_run(session, Listener())
        await feed(session, 16000, 180)
        run.done = run.playback_finished = True
        run.played_ms = int(run.emitted_ms)
        await feed(session, 16000, 8)
        assert session.overlap_run is None and not run.abort.is_set()
        assert session.metrics["barge_ins"] == 0 and not engine.calls
        await feed(session, 0, 10)
        await wait_for(lambda: len(engine.calls) == 1)
        assert np.count_nonzero(user_audio(engine.calls[0])[-1]) == 188 * 768

    asyncio.run(setup(check))


@pytest.mark.parametrize("action", ["adapt", "stop"])
def test_old_classifier_cannot_act_after_long_overlap_fallback(action):
    async def check(session, engine):
        listener = Listener(action, held=True)
        run = speaking_run(session, listener)
        previous = session.schedule_overlap(committed_overlap(session), run)
        await listener.entered.wait()
        await feed(session, 16000, 188)
        assert run.abort.is_set() and session.overlap_run is None
        listener.release.set()
        await previous
        assert not engine.calls and not session.semantic_pending
        assert not session.speech_paused and session.listening
        assert session.metrics["barge_ins"] == 1
        await feed(session, 0, 10)
        await wait_for(lambda: len(engine.calls) == 1)
        assert np.count_nonzero(user_audio(engine.calls[0])[-1]) == 188 * 768

    asyncio.run(setup(check))
