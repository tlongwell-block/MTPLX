"""Independent hold/race review: injected semantics, no models or audio device."""
import asyncio
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from test_frankie_background_session import call, output, setup, wait_for
from test_frankie_semantic_session import Listener as BaseListener, completed_run, enable, feed


class Listener(BaseListener):
    async def classify(self, *args, **kwargs):
        return await asyncio.wait_for(super().classify(*args, **kwargs), timeout=2)


class Vad:
    def __call__(self, frame):
        return float(np.max(np.abs(frame)) > .1)

    def reset(self):
        pass


def semantic(s, listener):
    enable(s, listener)
    s.vad = Vad()
    s.settings["output_modalities"] = ["audio"]
    s.settings["playback_feedback"] = True
    s.settings["playback_pause"] = True


def events(s, kind):
    return [row for row in s.outgoing._queue if row["type"] == kind]


def test_three_consecutive_voiced_frames_hold_once_without_aborting_or_rewriting():
    async def check(s, engine):
        semantic(s, Listener(held=True))
        run = completed_run(s)
        text = run.item["content"][0]["transcript"]
        await feed(s, 16000, 2)
        assert not events(s, "frankie.playback.pause")
        await feed(s, 16000, 1)
        assert [event["response_id"] for event in events(s, "frankie.playback.pause")] == [run.id]
        await feed(s, 16000, 3)
        assert len(events(s, "frankie.playback.pause")) == 1
        assert not run.abort.is_set() and not events(s, "frankie.playback.clear")
        assert run.item["content"][0]["transcript"] == text
        assert run.played_ms == 500 and run.emitted_ms == 1500
    asyncio.run(setup(check))


@pytest.mark.parametrize("action", ["continue", "wait"])
def test_valid_noninterrupting_decision_resumes_completed_but_undrained_response(action):
    async def check(s, engine):
        listener = Listener(action, held=True)
        semantic(s, listener)
        run = completed_run(s)
        await feed(s, 16000, 3)
        await feed(s, 0, 10)
        await asyncio.wait_for(listener.entered.wait(), timeout=2)
        assert not events(s, "frankie.playback.resume")
        listener.release.set()
        await wait_for(lambda: not s.semantic_pending)
        assert [event["response_id"] for event in events(s, "frankie.playback.resume")] == [run.id]
        assert not run.abort.is_set() and not engine.calls
        assert not run.playback_finished
    asyncio.run(setup(check))


def test_older_continue_cannot_resume_newer_speech_on_the_same_response():
    async def check(s, engine):
        listener = Listener("continue", held=True)
        semantic(s, listener)
        run = completed_run(s)
        await feed(s, 16000, 3)
        await feed(s, 0, 10)
        await asyncio.wait_for(listener.entered.wait(), timeout=2)
        await feed(s, 16000, 3)
        listener.release.set()
        await wait_for(lambda: not s.semantic_pending)
        assert s.listening and not run.abort.is_set()
        assert not events(s, "frankie.playback.resume") and not engine.calls
    asyncio.run(setup(check))


def test_input_clear_releases_hold_and_invalidates_inflight_semantics():
    async def check(s, engine):
        listener = Listener("adapt", held=True)
        semantic(s, listener)
        run = completed_run(s)
        await feed(s, 16000, 3)
        await feed(s, 0, 10)
        await asyncio.wait_for(listener.entered.wait(), timeout=2)
        await s.handle({"type": "input_audio_buffer.clear"})
        assert [event["response_id"] for event in events(s, "frankie.playback.resume")] == [run.id]
        listener.release.set()
        await wait_for(lambda: not s.semantic_pending)
        assert len(events(s, "frankie.playback.resume")) == 1
        assert not events(s, "frankie.playback.clear") and not run.abort.is_set()
        assert not engine.calls and not s.listening
    asyncio.run(setup(check))


def test_manual_cancel_clears_hold_and_stale_continue_never_resumes():
    async def check(s, engine):
        listener = Listener("continue", held=True)
        semantic(s, listener)
        run = completed_run(s)
        await feed(s, 16000, 3)
        await feed(s, 0, 10)
        await asyncio.wait_for(listener.entered.wait(), timeout=2)
        await s.handle({"type": "response.cancel", "response_id": run.id})
        listener.release.set()
        await wait_for(lambda: not s.semantic_pending)
        assert run.abort.is_set() and events(s, "frankie.playback.clear")
        assert not events(s, "frankie.playback.resume") and not engine.calls
    asyncio.run(setup(check))


def test_held_background_result_still_waits_for_playback_finished_after_resume():
    async def check(s, engine):
        listener = Listener("continue", held=True)
        semantic(s, listener)
        await call(s)
        run = completed_run(s)
        await feed(s, 16000, 3)
        await output(s, value="RESULT_SHOULD_WAIT")
        await s.handle({"type": "response.create"})
        await feed(s, 0, 10)
        await asyncio.wait_for(listener.entered.wait(), timeout=2)
        listener.release.set()
        await wait_for(lambda: not s.semantic_pending)
        assert events(s, "frankie.playback.resume")
        assert not engine.calls and s.queued_task_response
        engine.releases[0].set()
        await s.handle({"type": "frankie.playback.finished", "response_id": run.id,
                        "item_id": run.item_id, "audio_end_ms": 1500})
        await wait_for(lambda: len(engine.calls) == 1)
        assert "RESULT_SHOULD_WAIT" in str(engine.calls[0])
    asyncio.run(setup(check))


def test_pause_is_not_enabled_in_observation_mode():
    async def check(s, engine):
        semantic(s, Listener(held=True))
        s.settings["interruption_policy"] = "observe"
        completed_run(s)
        await feed(s, 16000, 4)
        assert not events(s, "frankie.playback.pause")
    asyncio.run(setup(check))


def test_actual_worklet_playback_hold_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js required for actual worklet regression")
    subprocess.run([node, "--test", str(Path(__file__).with_name("frankie_playback_hold_review.test.mjs"))],
                   check=True, timeout=30)


def test_hold_requires_consecutive_voiced_frames():
    async def check(s, engine):
        semantic(s, Listener(held=True))
        completed_run(s)
        await feed(s, 16000, 2)
        await feed(s, 0, 1)
        await feed(s, 16000, 2)
        assert not events(s, "frankie.playback.pause")
        await feed(s, 16000, 1)
        assert len(events(s, "frankie.playback.pause")) == 1
    asyncio.run(setup(check))


@pytest.mark.parametrize("disabled", ["playback_pause", "playback_feedback", "interrupt_response"])
def test_pause_requires_explicit_client_support_and_interrupt_enabled(disabled):
    async def check(s, engine):
        semantic(s, Listener(held=True))
        if disabled == "interrupt_response":
            s.settings["turn_detection"][disabled] = False
        else:
            s.settings[disabled] = False
        completed_run(s)
        await feed(s, 16000, 4)
        assert not events(s, "frankie.playback.pause")
    asyncio.run(setup(check))


@pytest.mark.parametrize("action,error", [("stop", None), ("adapt", None), ("yield", None),
                                          ("continue", TimeoutError("Classifier timeout"))])
def test_held_relinquish_or_error_clears_once_and_cannot_resume(action, error):
    async def check(s, engine):
        listener = Listener(action, error=error, held=True)
        semantic(s, listener)
        run = completed_run(s)
        await feed(s, 16000, 3)
        await feed(s, 0, 10)
        await asyncio.wait_for(listener.entered.wait(), timeout=2)
        engine.releases[0].set()
        listener.release.set()
        await wait_for(lambda: not s.semantic_pending)
        assert len(events(s, "frankie.playback.clear")) == 1
        assert not events(s, "frankie.playback.resume") and run.abort.is_set()
        assert not run.playback_paused
        if action == "stop":
            assert not engine.calls and s.speech_paused
        else:
            await wait_for(lambda: len(engine.calls) == 1)
            assert s.current is not run
    asyncio.run(setup(check))


@pytest.mark.parametrize("update", [
    {"frankie": {"playback_pause": False}},
    {"frankie": {"playback_feedback": False}},
    {"frankie": {"interruption_policy": "observe"}},
    {"turn_detection": None},
    {"turn_detection": {"type": "server_vad", "interrupt_response": False}},
])
def test_disabling_hold_support_resumes_an_existing_completed_tail(update):
    async def check(s, engine):
        listener = Listener("continue", held=True)
        semantic(s, listener)
        run = completed_run(s)
        await feed(s, 16000, 3)
        await feed(s, 0, 10)
        await asyncio.wait_for(listener.entered.wait(), timeout=2)
        await s.handle({"type": "session.update", "session": update})
        assert len(events(s, "frankie.playback.resume")) == 1
        assert not run.playback_paused
        listener.release.set()
        await wait_for(lambda: not s.semantic_pending)
        assert len(events(s, "frankie.playback.resume")) == 1
    asyncio.run(setup(check))


def test_input_clear_resets_voiced_confirmation_before_the_next_hold():
    async def check(s, engine):
        semantic(s, Listener(held=True))
        run = completed_run(s)
        await feed(s, 16000, 3)
        assert len(events(s, "frankie.playback.pause")) == 1
        await s.handle({"type": "input_audio_buffer.clear"})
        assert not run.playback_paused and s.voice_run == 0
        await feed(s, 16000, 2)
        assert len(events(s, "frankie.playback.pause")) == 1
        await feed(s, 16000, 1)
        assert len(events(s, "frankie.playback.pause")) == 2
    asyncio.run(setup(check))


def test_long_held_input_clears_once_and_preserves_the_full_user_turn():
    async def check(s, engine):
        semantic(s, Listener(held=True))
        run = completed_run(s)
        await feed(s, 16000, 187)  # 5984 ms remains inside the existing budget.
        assert run.playback_paused and not run.abort.is_set()
        assert len(events(s, "frankie.playback.pause")) == 1
        await feed(s, 16000, 1)  # 6016 ms falls back without dropping input.
        assert run.abort.is_set() and not run.playback_paused
        assert s.overlap_run is None and s.listening
        assert sum(map(len, s.frames)) == 188 * 768
        assert len(events(s, "frankie.playback.clear")) == 1
        assert not events(s, "frankie.playback.resume")
        await feed(s, 16000, 3)
        assert sum(map(len, s.frames)) == 191 * 768
        assert len(events(s, "frankie.playback.clear")) == 1
        assert not engine.calls
    asyncio.run(setup(check))
