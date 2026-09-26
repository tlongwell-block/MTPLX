"""Observation-only listening must still run through the normal VAD fast path."""

import asyncio

import pytest
from test_frankie_background_session import setup, wait_for
from test_frankie_semantic_session import Listener, completed_run, enable, feed


@pytest.mark.parametrize("action", ["continue", "adapt", "yield", "wait"])
def test_observer_sees_overlap_even_when_normal_vad_commits_speculation(action):
    async def check(session, engine):
        listener = Listener(action, held=True)
        enable(session, listener)
        session.settings["interruption_policy"] = "observe"
        previous = completed_run(session)
        previous.done = False
        observed = []
        classify = listener.classify

        async def record(item, run, *, fragments=None):
            observed.append((item, run))
            return await classify(item, run, fragments=fragments)

        listener.classify = record
        await feed(session, 16000, 4)
        assert previous.abort.is_set()  # Normal VAD behavior is retained.
        await feed(session, 0, 3)
        speculative = session.spec
        assert speculative is not None and not speculative.visible
        await feed(session, 0, 7)
        async with asyncio.timeout(2):
            await listener.entered.wait()
        assert session.spec is None and speculative.visible
        assert observed == [(speculative.input, previous)]
        listener.release.set()
        await wait_for(lambda: not session.semantic_pending)
        events = []
        while not session.outgoing.empty():
            events.append(session.outgoing.get_nowait())
        decisions = [event for event in events
                     if event["type"] == "frankie.interaction.observed"]
        assert len(decisions) == 1 and decisions[0]["action"] == action
        assert decisions[0]["apply"] is False
        assert not speculative.input.get("_listener_consumed")
        assert session.current is speculative and not speculative.abort.is_set()
        assert session.metrics["speculations"] == 1
        assert session.metrics["barge_ins"] == 1

    asyncio.run(setup(check))
