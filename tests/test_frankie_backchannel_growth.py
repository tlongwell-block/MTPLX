"""Growing acoustic evidence is usable only for the fast verbal-nod policy."""
import asyncio

import pytest
from test_frankie_background_session import setup, wait_for
from test_frankie_fast_turns_review import FastListener, fast
from test_frankie_prefix_adapter import snapshot
from test_frankie_regeneration import events, staged_reply
from test_frankie_streaming_session import clear_events, feed

from mtplx.frankie.listening import PrefixEvidence


@pytest.mark.parametrize(("previous", "current", "expected"), [
    ("Yeah but I", "Yeah but I want to change that", True),
    ("Actually", "Actually Thursday", True),
    ("Stop.", "Stop talking", True),
    ("YEAH, but", "yeah but I", True),
    ("Yeah", "Yeah", True),
    ("", "Actually Thursday", False),
    ("...", "Actually Thursday", False),
    ("Actually Thursday", "Actually", False),
    ("I hope", "I hear you", False),
    ("Stop", "Stopping", False),
])
def test_prefix_stability_requires_all_earlier_words(previous, current, expected):
    ticket = snapshot()
    evidence = PrefixEvidence(ticket, current, previous,
                              ticket.samples - ticket.sample_rate * 160 // 1000)
    assert evidence.stable_prefix(160) is expected
    assert not evidence.stable_prefix(161)
    if previous != "Yeah":
        assert not evidence.stable(160)  # Semantic policy still demands whole-text equality.
    assert evidence.text == current and evidence.previous_text == previous


def test_prefix_stability_requires_comparison_evidence():
    assert not PrefixEvidence(snapshot(), "Actually Thursday").stable_prefix(160)


class GrowingListener(FastListener):
    def __init__(self, previous, current):
        super().__init__(current)
        self.previous = previous

    async def hear_prefix(self, ticket):
        evidence, stats = await super().hear_prefix(ticket)
        return PrefixEvidence(ticket, evidence.text, self.previous, evidence.previous_samples), stats


@pytest.mark.parametrize(("previous", "current", "action"), [
    ("Yeah but I", "Yeah but I want to change that", "yield"),
    ("Actually", "Actually Thursday", "yield"),
    ("Stop.", "Stop talking", "yield"),
    ("Yeah", "Yeah but I", "wait"),
    ("I", "I want to change that", "wait"),
    ("Go", "Go away", "wait"),
    ("Im.", "I mean Thursday", "wait"),
    ("Yeah", "Yeah?", "wait"),
    ("I hope", "Mhm", "continue"),
    ("Yeah but", "Yeah", "continue"),
])
def test_fast_positive_requires_previous_and_current_full_text_to_be_non_nods(previous, current, action):
    async def check(s, engine):
        listener = GrowingListener(previous, current)
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        observed = next(event for event in events(s) if event["type"] == "frankie.interaction.prefix")
        assert observed["action"] == action and observed["apply"] is (action == "yield")
        assert observed["transcript"] == current and observed["previous_transcript"] == previous
        assert old.abort.is_set() is (action == "yield")
        assert not old.playback_paused and not engine.calls and not listener.rechecks
        assert s.listening  # Neither acoustic transcript owns the final user turn.
        if action != "yield":
            assert clear_events(s) == []
            assert s.backchannel_text == ("" if action == "wait" else current)
    asyncio.run(setup(check))
