"""Streaming listener plumbing, using controlled evidence without model inference.

These cases qualify lifecycle/ownership only, not the semantic accuracy or
latency of the real listener. Input travels through the actual VAD session path.
"""

import asyncio
import base64
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_frankie_background_session import call, output, setup, user, wait_for
from test_frankie_regeneration import events, staged_reply

from mtplx.frankie.listening import ListeningDecision, PrefixEvidence


class Vad:
    def __call__(self, frame):
        return float(np.max(np.abs(frame)) > .1)

    def reset(self):
        pass


class Listener:
    def __init__(self, action="yield", *, held=False, error=None, recheck_text=None,
                 qualified=True, final_action="yield", ignore_cancel=False):
        self.action, self.error, self.qualified = action, error, qualified
        self.recheck_text, self.final_action = recheck_text, final_action
        self.ignore_cancel = ignore_cancel
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.snapshots, self.contexts, self.rechecks, self.final_items = [], [], [], []
        self.cancelled = 0
        self.service = NS(warm_listener=dict)
        if not held:
            self.release.set()

    def cancel(self):
        self.cancelled += 1

    async def classify_prefix(self, ticket, **kwargs):
        self.snapshots.append(ticket)
        self.contexts.append(kwargs)
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                if not self.ignore_cancel:
                    raise
        if self.error:
            raise self.error
        text = "Yes, I understand." if self.action == "continue" else "Actually, change the date."
        evidence = PrefixEvidence(ticket, text, text, ticket.samples - 3840)
        decision = (ListeningDecision(self.action, True, True) if self.qualified
                    else NS(action=self.action))
        return decision, evidence, {"fixture": True}

    async def recheck_prefix(self, snapshot):
        self.rechecks.append(snapshot)
        return PrefixEvidence(snapshot, self.recheck_text or "Actually, change the date.")

    async def classify(self, item, run, *, fragments=None):
        self.final_items.append(item)
        for fragment in fragments or (item,):
            for part in fragment["content"]:
                if part["type"] == "input_audio":
                    part["_transcript"] = "Please change the date to Thursday."
        return NS(action=self.final_action), NS(user_text="Please change the date to Thursday."), {}


async def enable(s, listener, *, mode="semantic", interrupt=True):
    s.listener = listener
    await s.handle({"type": "session.update", "session": {
        "frankie": {"interruption_policy": "semantic", "streaming_listener": mode,
                    "playback_feedback": True, "playback_pause": False},
        "turn_detection": {"type": "server_vad", "threshold": .5,
                           "silence_duration_ms": 320, "create_response": True,
                           "interrupt_response": interrupt}}})
    s.vad = Vad()


async def feed(s, value, frames):
    raw = np.full(768, value, dtype="<i2")
    encoded = base64.b64encode(raw).decode()
    for _ in range(frames):
        await s.receive_audio(encoded)
        await asyncio.sleep(0)


def clear_events(s):
    return [event for event in events(s) if event["type"] in {
        "frankie.playback.clear", "frankie.playback.pause", "frankie.playback.resume"}]


@pytest.mark.parametrize("action", ["continue", "wait"])
def test_backchannel_or_wait_keeps_speech_playing_without_new_response(action):
    async def check(s, e):
        listener = Listener(action)
        await enable(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        assert not old.abort.is_set() and not old.playback_paused
        assert not e.calls and s.listening and len(s.frames) == 10
        assert listener.contexts[0] == {"heard_text": "The first point.",
                                        "pending_work": False, "assistant_name": "Frankie"}
        assert clear_events(s) == []
    asyncio.run(setup(check))


def test_prefix_yield_waits_for_complete_endpoint_and_retains_every_input_sample():
    async def check(s, e):
        listener = Listener()
        await enable(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: old.abort.is_set())
        assert s.listening and not e.calls
        assert old.item["_interrupted_draft"]["text"] == "The old plan."
        await feed(s, 17000, 6)
        await feed(s, 0, 9)
        await asyncio.sleep(.01)
        assert s.listening and s.spec is None and not e.calls
        await feed(s, 0, 1)
        await wait_for(lambda: len(e.calls) == 1)
        inputs = [item for item in e.calls[0] if item.get("role") == "user"]
        assert len(inputs) == 1 and not listener.final_items
        pcm = inputs[0]["content"][0]["_pcm"]
        expected = np.concatenate([np.full(7680, 16000), np.full(4608, 17000), np.zeros(7680)]) / 32768
        np.testing.assert_array_equal(pcm, expected)
        assert s.current is not old and not s.current.abort.is_set()
        assert len(clear_events(s)) == 1
    asyncio.run(setup(check))


@pytest.mark.parametrize("mode,qualified,interrupt", [
    ("observe", True, True), ("semantic", False, True), ("semantic", True, False)])
def test_unqualified_observation_only_or_disabled_interruption_never_controls(mode, qualified, interrupt):
    async def check(s, e):
        listener = Listener(qualified=qualified)
        await enable(s, listener, mode=mode, interrupt=interrupt)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        assert not old.abort.is_set() and not e.calls and clear_events(s) == []
    asyncio.run(setup(check))


def test_changed_transcript_recheck_cannot_apply_an_obsolete_prefix_decision():
    async def check(s, e):
        listener = Listener(held=True, recheck_text="Actually, don't change the date.")
        await enable(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await listener.entered.wait()
        await feed(s, 16000, 2)
        listener.release.set()
        await wait_for(lambda: s.prefix_task is None)
        assert len(listener.rechecks) == 1
        assert not old.abort.is_set() and clear_events(s) == []
        assert not e.calls
    asyncio.run(setup(check))


def test_cancelled_prefix_cannot_clear_a_replacement_even_if_backend_returns_late():
    async def check(s, e):
        listener = Listener(held=True, ignore_cancel=True)
        await enable(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await listener.entered.wait()
        await s.handle({"type": "input_audio_buffer.clear"})
        await user(s, "A different question.")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        replacement = s.current
        clear_events(s)
        listener.release.set()
        await wait_for(lambda: s.prefix_task is None)
        assert replacement is not old and not replacement.abort.is_set()
        assert clear_events(s) == []
        assert s.prefix_listener.inflight is None
    asyncio.run(setup(check))


def test_failed_prefix_preserves_the_full_utterance_for_final_handling():
    async def check(s, e):
        listener = Listener(error=RuntimeError("Fixture ear failure"))
        await enable(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        assert not old.abort.is_set() and clear_events(s) == []
        await feed(s, 0, 10)
        await wait_for(lambda: len(e.calls) == 1)
        assert old.abort.is_set() and len(listener.final_items) == 1
        pcm = listener.final_items[0]["content"][0]["_pcm"]
        assert len(pcm) == 20 * 768
        np.testing.assert_array_equal(pcm[:7680], np.full(7680, 16000 / 32768))
    asyncio.run(setup(check))


def test_tool_result_waits_for_the_user_after_prefix_yield_and_is_not_lost():
    async def check(s, e):
        listener = Listener()
        await enable(s, listener)
        await call(s)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: old.abort.is_set())
        assert listener.contexts[0]["pending_work"] is True
        await output(s, value="LATE_LOOKUP_RESULT")
        await s.handle({"type": "response.create"})
        await feed(s, 17000, 4)
        assert s.queued_task_response and not e.calls and s.listening
        await feed(s, 0, 10)
        await wait_for(lambda: len(e.calls) == 1)
        assert "LATE_LOOKUP_RESULT" not in str(e.calls[0])
        e.releases[0].set()
        await wait_for(lambda: len(e.calls) == 2)
        assert str(e.calls[1]).count("LATE_LOOKUP_RESULT") == 1
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        assert len(e.calls) == 2
    asyncio.run(setup(check))


def test_off_mode_never_submits_prefix_jobs():
    async def check(s, e):
        listener = Listener()
        await enable(s, listener, mode="off")
        old = staged_reply(s)
        await feed(s, 16000, 15)
        assert not listener.snapshots and s.prefix_task is None
        assert s.prefix_listener.retained_bytes == 0
        assert not old.abort.is_set() and clear_events(s) == []
    asyncio.run(setup(check))


def test_observer_audio_budget_does_not_acoustically_yield_or_discard_long_user_input():
    async def check(s, e):
        listener = Listener("continue", final_action="continue")
        await enable(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 200)
        assert s.listening and sum(len(frame) for frame in s.frames) == 200 * 768
        assert s.prefix_listener.retained_bytes <= 6 * 24000 * 2
        assert not old.abort.is_set() and not e.calls and clear_events(s) == []
        await feed(s, 0, 10)
        await wait_for(lambda: bool(listener.final_items))
        assert len(listener.final_items[0]["content"][0]["_pcm"]) == 210 * 768
        assert not old.abort.is_set()
    asyncio.run(setup(check))


def test_manual_response_cancel_during_prefix_keeps_accepting_the_rest_of_user_audio():
    async def check(s, e):
        listener = Listener(held=True)
        await enable(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await listener.entered.wait()
        await s.handle({"type": "response.cancel", "response_id": old.id})
        await feed(s, 17000, 3)
        assert s.listening and sum(len(frame) for frame in s.frames) == 13 * 768
        assert old.abort.is_set() and not e.calls
        await wait_for(lambda: s.prefix_task is None)
        assert s.prefix_listener.inflight is None
    asyncio.run(setup(check))


def test_stale_explicit_create_cannot_start_partial_reply_after_prefix_yield_without_tasks():
    async def check(s, e):
        listener = Listener()
        await enable(s, listener)
        await s.handle({"type": "session.update", "session": {"frankie": {"background_tasks": False}}})
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: old.abort.is_set())
        await s.handle({"type": "response.create"})
        await asyncio.sleep(.01)
        assert s.listening and not e.calls and s.current is old
    asyncio.run(setup(check))


def test_prefix_cancelled_before_coroutine_starts_does_not_leak_its_reservation():
    async def check(s, e):
        listener = Listener("continue")
        await enable(s, listener)
        staged_reply(s)
        # One receive callback can reserve the prefix before the task has run.
        encoded = base64.b64encode(np.full(7680, 16000, dtype="<i2")).decode()
        await s.receive_audio(encoded)
        assert s.prefix_task is not None and s.prefix_listener.inflight is not None
        assert listener.snapshots == []
        await s.handle({"type": "input_audio_buffer.clear"})
        await asyncio.sleep(.01)
        assert s.prefix_task is None and s.prefix_listener.inflight is None
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
    asyncio.run(setup(check))


def test_full_endpoint_supersedes_a_slow_prefix_and_late_result_cannot_clear_new_answer():
    async def check(s, e):
        listener = Listener(held=True, ignore_cancel=True)
        await enable(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await listener.entered.wait()
        await feed(s, 0, 10)
        await wait_for(lambda: len(e.calls) == 1)
        replacement = s.current
        assert not s.listening and len(listener.final_items) == 1 and old.abort.is_set()
        clear_events(s)
        listener.release.set()
        await wait_for(lambda: s.prefix_task is None)
        assert not replacement.abort.is_set() and clear_events(s) == []
        assert s.prefix_listener.inflight is None and not s.semantic_pending
        assert len(e.calls) == 1
    asyncio.run(setup(check))
