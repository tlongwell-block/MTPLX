"""Use injected decisions to isolate semantic turn execution from perception."""

import asyncio
import base64
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_frankie_background_session import call, output, setup, user, wait_for

from mtplx.frankie.session import Response


class Listener:
    def __init__(self, action="continue", error=None, held=False):
        self.action, self.error = action, error
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = 0
        if not held:
            self.release.set()

    def cancel(self):
        self.cancelled += 1

    async def classify(self, item, run, *, fragments=None):
        self.entered.set()
        await self.release.wait()
        if self.error:
            raise self.error
        part = next(p for p in item["content"] if p["type"] == "input_audio")
        part["_transcript"] = (
            "Mm-hmm." if self.action == "continue" else "No, Thursday."
        )
        return NS(action=self.action), NS(user_text=part["_transcript"]), {}


def enable(s, listener):
    s.listener = listener
    s.settings["interruption_policy"] = "semantic"
    s.vad = lambda frame: float(np.max(np.abs(frame)) > 0.1)


def completed_run(s):
    run = Response(
        visible=True,
        done=True,
        emitted_ms=1500,
        played_ms=500,
        first_audio_at=time.monotonic(),
    )
    run.chunks = [
        {"text": "Heard.", "end_ms": 500},
        {"text": "Never heard.", "end_ms": 1500},
    ]
    run.item = {
        "id": run.item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_audio", "transcript": "Heard. Never heard."}],
    }
    s.current = run
    s.items.append(run.item)
    s.playback_runs[run.item_id] = run
    return run


def committed_overlap(s):
    item = s.audio_item(np.zeros(2400), speech_end_ms=100)
    s.items.append(item)
    s.accept_input(semantic_overlap=True)
    return item


async def feed(s, value, frames):
    audio = base64.b64encode(np.full(768, value, dtype="<i2")).decode()
    for _ in range(frames):
        await s.receive_audio(audio)
        await asyncio.sleep(0)


@pytest.mark.parametrize(
    "error", [None, RuntimeError("Probe failed"), TimeoutError("Probe timeout")]
)
def test_adaptation_and_failure_both_remove_unheard_text_before_replanning(error):
    async def check(s, e):
        enable(s, Listener("adapt", error))
        run = completed_run(s)
        item = committed_overlap(s)
        e.releases[0].set()
        task = s.schedule_overlap(item, run)
        assert s.semantic_pending == 1  # Reserved synchronously before execution.
        await task
        await wait_for(lambda: len(e.calls) == 1)
        assert e.calls[0][0]["content"][0]["transcript"] == "Heard."
        assert "Never heard." not in str(e.calls[0])
        assert run.abort.is_set() and not s.semantic_pending
        audio = next(p for p in item["content"] if p["type"] == "input_audio")
        assert audio.get("_listener_transcript") == ("No, Thursday." if error is None else None)

    asyncio.run(setup(check))


def test_new_uncommitted_speech_invalidates_an_older_adaptation_decision():
    async def check(s, e):
        listener = Listener("adapt", held=True)
        enable(s, listener)
        run = completed_run(s)
        item = committed_overlap(s)
        task = s.schedule_overlap(item, run)
        await listener.entered.wait()
        await feed(s, 16000, 1)  # New utterance begins before the old result arrives.
        listener.release.set()
        await task
        assert s.listening and not run.abort.is_set()
        assert not e.calls and not s.semantic_pending
        assert "_listener_transcript" not in item["content"][0]
        assert not item.get("_listener_consumed")

    asyncio.run(setup(check))


@pytest.mark.parametrize("error", [None, RuntimeError("Cancelled probe")])
def test_manual_cancel_cannot_be_followed_by_a_stale_semantic_restart(error):
    async def check(s, e):
        listener = Listener("adapt", error, held=True)
        enable(s, listener)
        run = completed_run(s)
        task = s.schedule_overlap(committed_overlap(s), run)
        await listener.entered.wait()
        await s.handle({"type": "response.cancel", "response_id": run.id})
        listener.release.set()
        await task
        assert not e.calls
        assert not s.semantic_pending

    asyncio.run(setup(check))


@pytest.mark.parametrize("action", ["continue", "wait"])
def test_acknowledgment_is_handled_without_a_followup_on_stale_response_create(action):
    async def check(s, e):
        enable(s, Listener(action))
        run = completed_run(s)
        await s.schedule_overlap(committed_overlap(s), run)
        assert not run.abort.is_set()
        await s.handle({"type": "response.create"})
        await asyncio.sleep(0.01)
        assert not e.calls

    asyncio.run(setup(check))


def test_continue_preserves_a_queued_tool_result_through_real_vad_commit():
    async def check(s, e):
        listener = Listener("continue", held=True)
        enable(s, listener)
        s.settings["playback_feedback"] = True
        await call(s)
        await user(s)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        run = s.current
        run.emitted_ms = 1500
        run.first_audio_at = time.monotonic()
        await feed(s, 16000, 4)
        await output(s)
        await s.handle({"type": "response.create"})
        assert s.queued_task_response
        await feed(s, 0, 10)
        await listener.entered.wait()
        assert s.semantic_pending == 1
        e.releases[0].set()
        await wait_for(lambda: run.done)
        assert len(e.calls) == 1  # Cannot start result reply before the decision.
        listener.release.set()
        await wait_for(lambda: not s.semantic_pending)
        assert len(e.calls) == 1  # The decision cannot bypass queued playback.
        await s.handle({
            "type": "frankie.playback.finished", "item_id": run.item_id,
            "response_id": run.id, "audio_end_ms": int(run.emitted_ms),
        })
        await wait_for(lambda: len(e.calls) == 2)
        assert "Background task notice" in str(e.calls[1])
        assert not run.abort.is_set()
        e.releases[1].set()
        await wait_for(lambda: s.current.done)

    asyncio.run(setup(check))


def test_result_arriving_after_generation_done_waits_for_listener_resolution():
    async def check(s, e):
        listener = Listener("continue", held=True)
        enable(s, listener)
        await call(s)
        run = completed_run(s)
        task = s.schedule_overlap(committed_overlap(s), run)
        await listener.entered.wait()
        await output(s)
        await s.handle({"type": "response.create"})
        assert s.queued_task_response and not e.calls
        listener.release.set()
        await task
        assert not e.calls  # A result also waits for the audio already queued.
        await s.handle({
            "type": "frankie.playback.finished", "item_id": run.item_id,
            "response_id": run.id, "audio_end_ms": int(run.emitted_ms),
        })
        await wait_for(lambda: len(e.calls) == 1)
        assert "Background task notice" in str(e.calls[0])

    asyncio.run(setup(check))


def test_result_before_listener_yield_waits_for_replacement_playback_then_runs_once():
    async def check(s, e):
        listener = Listener("yield", held=True)
        enable(s, listener)
        s.settings["playback_feedback"] = True
        await call(s)
        old = completed_run(s)
        item = committed_overlap(s)
        observation = s.schedule_overlap(item, old)
        await listener.entered.wait()
        await output(s, value="DELAYED_RESULT_42")
        result_item = s.items[-1]
        await s.handle({"type": "response.create"})
        listener.release.set()
        await observation
        await wait_for(lambda: len(e.calls) == 1)
        replacement = s.current
        assert "DELAYED_RESULT_42" not in str(e.calls[0])
        assert '"pending"' in str(e.calls[0])
        assert s.queued_task_response and s.unhandled_task_results == {"lookup_1"}
        assert not s.task_ledger.tasks["lookup_1"].consumed
        # The result remains at its actual arrival position in wire history.
        assert s.items.index(item) < s.items.index(result_item) < s.items.index(replacement.item)
        replacement.emitted_ms = 1000
        replacement.first_audio_at = time.monotonic()
        e.releases[0].set()
        await wait_for(lambda: replacement.done)
        assert len(e.calls) == 1
        await s.handle({
            "type": "frankie.playback.finished", "item_id": replacement.item_id,
            "response_id": replacement.id, "audio_end_ms": 1000,
        })
        await wait_for(lambda: len(e.calls) == 2)
        assert "DELAYED_RESULT_42" in str(e.calls[1])
        assert s.task_ledger.tasks["lookup_1"].consumed
        assert not s.queued_task_response and not s.unhandled_task_results
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        await asyncio.sleep(0.01)
        assert len(e.calls) == 2

    asyncio.run(setup(check))


def test_result_arriving_during_replacement_keeps_snapshot_and_followup():
    async def check(s, e):
        enable(s, Listener("yield"))
        await call(s)
        old = completed_run(s)
        await s.schedule_overlap(committed_overlap(s), old)
        await wait_for(lambda: len(e.calls) == 1)
        await output(s, value="ARRIVED_DURING_REPLY")
        await s.handle({"type": "response.create"})
        assert "ARRIVED_DURING_REPLY" not in str(e.calls[0])
        assert s.queued_task_response
        e.releases[0].set()
        await wait_for(lambda: len(e.calls) == 2)
        assert "ARRIVED_DURING_REPLY" in str(e.calls[1])
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        assert len(e.calls) == 2

    asyncio.run(setup(check))


@pytest.mark.parametrize("cancel_replacement", [False, True])
def test_fresh_user_query_can_consume_reserved_result_without_an_extra_followup(cancel_replacement):
    async def check(s, e):
        listener = Listener("yield", held=True)
        enable(s, listener)
        await call(s)
        old = completed_run(s)
        observation = s.schedule_overlap(committed_overlap(s), old)
        await listener.entered.wait()
        await output(s, value="AVAILABLE_FOR_FRESH_QUERY")
        await s.handle({"type": "response.create"})
        listener.release.set()
        await observation
        await wait_for(lambda: len(e.calls) == 1)
        replacement = s.current
        assert "AVAILABLE_FOR_FRESH_QUERY" not in str(e.calls[0])
        if cancel_replacement:
            await s.handle({"type": "response.cancel", "response_id": replacement.id})
            assert not s.queued_task_response
            assert s.unhandled_task_results == {"lookup_1"}
        await user(s, "What was the lookup result?")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 2)
        assert replacement.abort.is_set()
        assert "AVAILABLE_FOR_FRESH_QUERY" in str(e.calls[1])
        assert e.calls[1][-1]["content"][0]["text"] == "What was the lookup result?"
        assert not s.unhandled_task_results and not s.queued_task_response
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        await asyncio.sleep(0.01)
        assert len(e.calls) == 2

    asyncio.run(setup(check))


def test_superseded_reserved_result_is_never_presented_after_replacement():
    async def check(s, e):
        listener = Listener("yield", held=True)
        enable(s, listener)
        await call(s)
        old = completed_run(s)
        observation = s.schedule_overlap(committed_overlap(s), old)
        await listener.entered.wait()
        await output(s, value="OBSOLETE_RESERVED_RESULT")
        await s.handle({"type": "response.create"})
        listener.release.set()
        await observation
        await wait_for(lambda: len(e.calls) == 1)
        await s.handle({"type": "frankie.task.cancel", "call_id": "lookup_1", "reason": "superseded"})
        assert not s.queued_task_response and not s.unhandled_task_results
        e.releases[0].set()
        await wait_for(lambda: s.current.done)
        assert len(e.calls) == 1
        await user(s, "What is next?")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 2)
        assert "OBSOLETE_RESERVED_RESULT" not in str(e.calls[1])

    asyncio.run(setup(check))


def test_nested_fresh_semantic_input_can_consume_the_now_available_result():
    async def check(s, e):
        listener = Listener("yield", held=True)
        enable(s, listener)
        await call(s)
        old = completed_run(s)
        observation = s.schedule_overlap(committed_overlap(s), old)
        await listener.entered.wait()
        await output(s, value="NESTED_RESERVED_RESULT")
        await s.handle({"type": "response.create"})
        listener.release.set()
        await observation
        await wait_for(lambda: len(e.calls) == 1)
        first_replacement = s.current
        first_replacement.emitted_ms = 1500
        first_replacement.played_ms = 500
        first_replacement.first_audio_at = time.monotonic()
        await feed(s, 16000, 4)
        await feed(s, 0, 10)
        await wait_for(lambda: len(e.calls) == 2)
        assert first_replacement.abort.is_set()
        assert "NESTED_RESERVED_RESULT" not in str(e.calls[0])
        assert "NESTED_RESERVED_RESULT" in str(e.calls[1])
        assert not s.queued_task_response and not s.unhandled_task_results
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        await asyncio.sleep(0.01)
        assert len(e.calls) == 2

    asyncio.run(setup(check))


def test_result_already_queued_before_semantic_speech_is_available_in_its_answer():
    async def check(s, e):
        listener = Listener("yield", held=True)
        enable(s, listener)
        s.settings["playback_feedback"] = True
        await call(s)
        completed_run(s)
        await output(s, value="KNOWN_BEFORE_SPEECH")
        await s.handle({"type": "response.create"})
        assert s.queued_task_response
        await feed(s, 16000, 4)
        await feed(s, 0, 10)
        await listener.entered.wait()
        item = s.items[-1]
        assert item["_task_results_at_capture"] == ("lookup_1",)
        listener.release.set()
        await wait_for(lambda: len(e.calls) == 1)
        assert "KNOWN_BEFORE_SPEECH" in str(e.calls[0])
        assert not s.queued_task_response and not s.unhandled_task_results
        e.releases[0].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        assert len(e.calls) == 1

    asyncio.run(setup(check))


def test_grouped_speech_uses_result_availability_at_its_earliest_fragment():
    async def check(s, e):
        listener = Listener("wait")
        enable(s, listener)
        await call(s)
        old = completed_run(s)
        first = committed_overlap(s)
        first.update(_speech_start_ms=100, _speech_end_ms=200)
        await s.schedule_overlap(first, old)
        await output(s, value="ARRIVED_BETWEEN_FRAGMENTS")
        await s.handle({"type": "response.create"})
        s.capture_task_results = ("lookup_1",)
        second = committed_overlap(s)
        second.update(_speech_start_ms=250, _speech_end_ms=350)
        listener.action = "yield"
        await s.schedule_overlap(second, old)
        await wait_for(lambda: len(e.calls) == 1)
        assert "ARRIVED_BETWEEN_FRAGMENTS" not in str(e.calls[0])
        assert s.queued_task_response and s.unhandled_task_results == {"lookup_1"}
        assert first["_listener_consumed"] and second["_listener_consumed"]
        e.releases[0].set()
        await wait_for(lambda: len(e.calls) == 2)
        assert "ARRIVED_BETWEEN_FRAGMENTS" in str(e.calls[1])

    asyncio.run(setup(check))


@pytest.mark.parametrize("during_tentative", [False, True])
def test_ordinary_vad_commit_preserves_result_arriving_after_audio_capture(during_tentative):
    async def check(s, e):
        s.vad = lambda frame: float(np.max(np.abs(frame)) > 0.1)
        await call(s)
        await feed(s, 16000, 4)
        if during_tentative:
            await feed(s, 0, 3)
            await wait_for(lambda: len(e.calls) == 1)
            assert s.spec is not None and not s.spec.visible
        await output(s, value="AFTER_CAPTURE_RESULT")
        await s.handle({"type": "response.create"})
        assert s.queued_task_response
        await feed(s, 0, 7 if during_tentative else 10)
        await wait_for(lambda: len(e.calls) == 1 and s.current.visible)
        assert "AFTER_CAPTURE_RESULT" not in str(e.calls[0])
        assert s.queued_task_response and s.unhandled_task_results == {"lookup_1"}
        e.releases[0].set()
        await wait_for(lambda: len(e.calls) == 2)
        assert "AFTER_CAPTURE_RESULT" in str(e.calls[1])
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        assert len(e.calls) == 2

    asyncio.run(setup(check))


def test_new_typed_input_blocks_automatic_result_reply_until_response_requested():
    async def check(s, e):
        await call(s)
        await user(s)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        old = s.current
        await output(s, value="AVAILABLE_AFTER_TYPED_INPUT")
        await s.handle({"type": "response.create"})
        await user(s, "What was the lookup result?")
        await wait_for(lambda: old.done)
        await asyncio.sleep(0.02)
        assert len(e.calls) == 1
        assert s.queued_task_response and s.unhandled_task_results == {"lookup_1"}
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 2)
        assert "AVAILABLE_AFTER_TYPED_INPUT" in str(e.calls[1])
        assert not s.queued_task_response and not s.unhandled_task_results

    asyncio.run(setup(check))


def test_fresh_user_response_takes_priority_over_old_playback_tail_and_result_queue():
    async def check(s, e):
        await call(s)
        old = completed_run(s)
        s.settings["playback_feedback"] = True
        await output(s, value="KNOWN_FRESH_RESULT")
        await s.handle({"type": "response.create"})
        assert s.queued_task_response
        await user(s, "What was the lookup result?")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        assert "KNOWN_FRESH_RESULT" in str(e.calls[0])
        assert old.abort.is_set() and not s.queued_task_response

    asyncio.run(setup(check))


def test_manual_committed_audio_retains_capture_when_result_arrives_after_item():
    async def check(s, e):
        await call(s)
        s.settings["turn_detection"] = None
        await feed(s, 16000, 4)
        await s.handle({"type": "input_audio_buffer.commit"})
        audio_item = s.items[-1]
        await output(s, value="AFTER_MANUAL_COMMIT")
        assert s.items[-1]["type"] == "function_call_output"
        # This single request both answers the committed user and reserves the
        # arriving result; the captured input is not necessarily the last item.
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        assert s.current.input is audio_item
        assert "AFTER_MANUAL_COMMIT" not in str(e.calls[0])
        assert s.queued_task_response
        e.releases[0].set()
        await wait_for(lambda: len(e.calls) == 2)
        assert "AFTER_MANUAL_COMMIT" in str(e.calls[1])
        assert s.current.input is None  # Result followup cannot reuse old capture.

    asyncio.run(setup(check))


def test_no_audio_yet_retains_normal_cancellation_in_semantic_mode():
    async def check(s, e):
        listener = Listener()
        enable(s, listener)
        await user(s)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        run = s.current
        assert not run.emitted_ms
        await feed(s, 16000, 3)
        assert run.abort.is_set()
        assert not listener.entered.is_set()

    asyncio.run(setup(check))


def test_listener_is_warmed_before_policy_activation_and_identity_is_validated():
    async def check(s, e):
        warmed = []
        s.listener = NS(
            cancel=lambda: None, service=NS(warm_listener=lambda: warmed.append(True))
        )
        await s.handle(
            {
                "type": "session.update",
                "session": {
                    "frankie": {
                        "interruption_policy": "semantic",
                        "assistant_name": "Companion",
                    }
                },
            }
        )
        assert warmed == [True]
        assert s.info()["frankie"]["assistant_name"] == "Companion"
        for name in ("", " " * 5, "x" * 65, True):
            with pytest.raises(ValueError, match="assistant_name"):
                await s.handle(
                    {
                        "type": "session.update",
                        "session": {"frankie": {"assistant_name": name}},
                    }
                )

    asyncio.run(setup(check))
