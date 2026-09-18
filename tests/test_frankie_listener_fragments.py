"""Fragmented VAD endpoints retain bounded, unconsumed semantic evidence."""

import asyncio
import json
from threading import Event

import numpy as np
import pytest
from test_frankie_background_session import setup, wait_for
from test_frankie_long_overlap import user_audio
from test_frankie_semantic_session import completed_run, enable, feed

from mtplx.frankie.perception import RealtimeListener
from mtplx.frankie.session import public


class Probe:
    def __init__(self, service, data, prepare, action, held):
        self.service, self.data, self.prepare = service, data, prepare
        self.action, self.held = action, held
        self.cancelled, self.ready = Event(), Event()
        self.internal = True

    async def receive(self):
        try:
            self.prepare(self)
            self.service.inputs.append(json.loads(self.data["messages"][-1]["content"]))
            while self.held and not self.cancelled.is_set():
                await asyncio.sleep(0.001)
            if self.cancelled.is_set():
                raise RuntimeError("Cancelled probe")
            return {"finish_reason": "stop", "message": {"content": self.action}}
        finally:
            self.service.jobs.discard(self)


class Service:
    def __init__(self, actions, *, hold_first=False):
        self.actions = iter(actions)
        self.hold_first = hold_first
        self.jobs, self.inputs = set(), []

    def submit(self, data, chat, *, internal, prepare):
        assert chat and internal
        job = Probe(self, data, prepare, next(self.actions), self.hold_first)
        self.hold_first = False
        self.jobs.add(job)
        return job


def prepare_listener(session, engine, texts, actions, *, hold_first=False):
    words = iter(texts)
    ear_calls = []

    def hear(pcm, rate):
        ear_calls.append((len(pcm), rate))
        return np.ones((1, 2)), next(words)

    engine.audio = type("Ear", (), {"hear": staticmethod(hear)})()
    service = Service(actions, hold_first=hold_first)
    enable(session, RealtimeListener(session, service))
    session.settings["playback_feedback"] = True
    run = completed_run(session)
    run.emitted_ms = 20000
    return run, service, ear_calls


def audio_items(session):
    return [item for item in session.items if item.get("role") == "user"]


async def utterance(session, frames=20):
    await feed(session, 16000, frames)
    await feed(session, 0, 10)


@pytest.mark.parametrize("first_action,held", [("adapt", True), ("wait", False)])
def test_cancelled_or_waiting_fragment_is_joined_and_grounded_once(first_action, held):
    async def check(session, engine):
        phrases = ["Make it an indoor picnic instead,", "because it might rain."]
        run, service, calls = prepare_listener(
            session, engine, phrases, [first_action, "adapt"], hold_first=held)
        await utterance(session)
        await wait_for(lambda: len(service.inputs) == 1)
        if not held:
            await wait_for(lambda: not session.semantic_pending)
        first = audio_items(session)[0]
        assert not first.get("_listener_consumed")
        await utterance(session, 10)
        await wait_for(lambda: len(engine.calls) == 1)
        await wait_for(lambda: not session.semantic_pending)
        assert [value["user_prefix"] for value in service.inputs] == [phrases[0], " ".join(phrases)]
        assert len(calls) == 2  # Completed ear work survives classifier cancellation.
        assert run.abort.is_set() and session.metrics["barge_ins"] == 1
        items = audio_items(session)
        assert all(item["_listener_consumed"] for item in items)
        for item, text in zip(items, phrases, strict=True):
            part = item["content"][0]
            assert part["_listener_transcript"] == text and "_rows" in part
            assert "_listener_transcript" not in public(item)["content"][0]
        events = list(session.outgoing._queue)
        captions = [event["item_id"] for event in events
                    if event["type"] == "conversation.item.input_audio_transcription.completed"]
        assert captions == [item["id"] for item in items]
        assert len(user_audio(engine.calls[0])) == 2  # No history/audio item was dropped.

    asyncio.run(setup(check))


@pytest.mark.parametrize("barrier", ["consumed", "time_gap", "new_response", "typed_input"])
def test_unrelated_or_consumed_fragments_are_not_reused(barrier):
    async def check(session, engine):
        phrases = ["Mm-hmm.", "Because it might rain."]
        run, service, calls = prepare_listener(
            session, engine, phrases, ["continue" if barrier == "consumed" else "wait", "wait"])
        await utterance(session)
        await wait_for(lambda: not session.semantic_pending)
        if barrier == "time_gap":
            await feed(session, 0, 50)
        elif barrier == "new_response":
            run = completed_run(session)
            run.emitted_ms = 20000
        elif barrier == "typed_input":
            # A new user message is a semantic boundary even within the gap.
            session.items.append({"type": "message", "role": "user", "content": []})
        await utterance(session, 10)
        await wait_for(lambda: not session.semantic_pending)
        assert [value["user_prefix"] for value in service.inputs] == phrases
        assert len(calls) == 2 and not run.abort.is_set() and not engine.calls

    asyncio.run(setup(check))


def test_fragment_budget_yields_at_boundary_and_preserves_all_audio_and_future_turns():
    async def check(session, engine):
        run, service, calls = prepare_listener(session, engine, ["Actually,"], ["wait"])
        await utterance(session, 100)  # 3.520 seconds, including endpoint padding.
        await wait_for(lambda: not session.semantic_pending)
        await feed(session, 16000, 67)  # 2.464 seconds with preroll; 5.984 total.
        assert not run.abort.is_set() and session.overlap_run is run
        await feed(session, 16000, 1)  # Combined buffer crosses 6.000 seconds.
        assert run.abort.is_set() and session.overlap_run is None
        assert session.listening and not engine.calls
        await feed(session, 16000, 10)
        assert session.metrics["barge_ins"] == 1
        await feed(session, 0, 10)
        await wait_for(lambda: len(engine.calls) == 1)
        captured = user_audio(engine.calls[0])
        assert len(captured) == 2
        assert sum(np.count_nonzero(pcm) for pcm in captured) == 178 * 768
        assert len(service.inputs) == len(calls) == 1
        assert not session.semantic_pending and not session.listening
        engine.releases[0].set()
        await wait_for(lambda: session.current.done)
        await utterance(session, 4)
        await wait_for(lambda: len(engine.calls) == 2)
        assert np.count_nonzero(user_audio(engine.calls[1])[-1]) == 4 * 768

    asyncio.run(setup(check))


def test_classifier_rejects_oversized_combined_audio_before_owner_work():
    async def check(session, engine):
        run, service, calls = prepare_listener(session, engine, [], [])
        fragments = tuple(session.audio_item(np.zeros(4 * 24000)) for _ in range(2))
        with pytest.raises(ValueError, match="six-second acoustic budget"):
            await session.listener.classify(fragments[-1], run, fragments=fragments)
        assert not calls and not service.jobs and not service.inputs

    asyncio.run(setup(check))
