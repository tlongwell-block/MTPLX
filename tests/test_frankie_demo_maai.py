import asyncio
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mtplx.frankie.demo.maai import FEATURES, Gate, _maybe_yield, install


def test_gate_yields_on_sustained_low_probability():
    gate = Gate()
    decisions = [gate.observe(.01, end, True) for end in (80, 160, 240)]
    assert decisions == [False, True, True]


def test_gate_holds_through_a_nod_then_releases_on_the_low_threshold():
    gate = Gate()
    # A nod (p >= .2 twice) arms the lower release threshold of .05.
    assert [gate.observe(p, end, True) for p, end in ((.5, 80), (.4, 160))] == [False, False]
    assert gate.seen_bc
    assert gate.observe(.1, 240, True) is False  # under .2, not under .05
    assert [gate.observe(.01, end, True) for end in (320, 400)] == [False, True]


def test_gate_ignores_stale_frames_and_silence_resets_runs():
    gate = Gate()
    assert gate.observe(.01, 80, True) is False
    assert gate.observe(.01, 80, True) is False  # same frame again
    assert gate.observe(.01, 160, False) is False
    assert gate.low_run == gate.high_run == 0
    assert gate.observe(.01, 240, True) is False


def session(**values):
    calls = []
    s = NS(listening=True, speech_start_ms=0, clock_ms=300, speech_id="u1", voice_run=3,
           overlap_run="run", maai_gate_args=(.2, .05, 2),
           event=lambda name, **v: calls.append(name),
           yield_prefix=lambda run, reason: calls.append(("yield", run, reason)))
    s.__dict__.update(values)
    return s, calls


def feed(s, p, end):
    s.maai_latest = dict(p=[p], audio_end_ms=end)
    s.clock_ms = end + 20
    _maybe_yield(s)


def test_confirmed_speech_yields_the_overlapped_reply():
    s, calls = session()
    feed(s, .01, 80)
    feed(s, .01, 160)
    assert ("yield", "run", "bench_acoustic_confirmed_non_backchannel") in calls


def test_short_voicing_never_yields():
    s, calls = session(voice_run=2)
    for end in (80, 160, 240):
        feed(s, .01, end)
    assert not [c for c in calls if c[0] == "yield"]


def test_late_or_pre_utterance_results_are_ignored():
    s, calls = session(speech_start_ms=500)
    feed(s, .01, 400)  # before this utterance began
    s.maai_latest = dict(p=[.01], audio_end_ms=600)
    s.clock_ms = 900  # result is 300 ms stale
    _maybe_yield(s)
    assert calls == []


def test_a_new_utterance_gets_a_fresh_gate():
    s, _ = session()
    feed(s, .5, 80)
    first = s.maai_gate
    s.speech_id = "u2"
    feed(s, .01, 160)
    assert s.maai_gate is not first


def test_install_forces_demo_settings_and_preserves_overlaps(monkeypatch):
    from mtplx.frankie.demo import maai
    from mtplx.frankie.session import Session

    for name in ("__init__", "handle", "info", "close", "append_prefix", "schedule_overlap"):
        monkeypatch.setattr(Session, name, getattr(Session, name))
    seen = []

    async def handle(self, event):
        seen.append(event)
    monkeypatch.setattr(Session, "handle", handle)

    class Worker:
        def __init__(self, session, detector):
            self.detector = detector
    monkeypatch.setattr(maai, "Worker", Worker)
    install("detector")

    events = []
    s = Session.__new__(Session)
    s.maai_worker, s.turn = None, "turn"
    s.event = lambda name, **v: events.append((name, v))
    asyncio.run(Session.handle(s, {"type": "session.update",
                                   "session": {"frankie": {"playback_pause": True, "x": 1}}}))
    assert s.maai_worker.detector == "detector" and s.turn.turn == "turn"
    assert seen[0]["session"]["frankie"] == {"x": 1, **FEATURES}
    Session.schedule_overlap(s, {"id": "item"}, "run")
    assert events == [("bench.backchannel_preserved", {"item_id": "item"})]


def test_detector_needs_maai(monkeypatch):
    import importlib.util
    import sys
    from mtplx.frankie.demo import maai

    monkeypatch.delitem(sys.modules, "maai", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="maai"):
        maai._import_bc_det()
