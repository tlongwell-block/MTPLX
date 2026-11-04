import asyncio
import base64
import json
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mtplx.frankie.demo import backchannels
from mtplx.frankie.demo.backchannels import GAP_MS, Takes, due


def session(**values):
    s = NS(listening=True, speech_start_ms=0, clock_ms=3000, backchannel_at=-GAP_MS,
           audible_response=lambda: None)
    s.__dict__.update(values)
    return s


def reading(end_ms=2950, p_sys=0.05, p_bc=0.6):
    return (end_ms * 16, p_sys, p_bc)


def test_fires_mid_turn_when_the_model_expects_a_listener_sound():
    assert due(session(), reading())


@pytest.mark.parametrize("change, value", [
    ("p_bc", 0.39),  # not a backchannel moment
    ("p_sys", 0.15),  # the turn may be passing to Frankie
    ("end_ms", 2600),  # reading too old for now
    ("end_ms", 1400),  # too early in the user's turn
])
def test_holds_when_the_reading_says_no(change, value):
    assert not due(session(), reading(**{change: value}))


def test_holds_outside_a_user_turn_over_a_reply_and_between_sounds():
    assert not due(session(listening=False), reading())
    assert not due(session(audible_response=lambda: "run"), reading())
    assert not due(session(backchannel_at=3000 - GAP_MS + 1), reading())
    assert due(session(backchannel_at=3000 - GAP_MS), reading())
    assert not due(session(), None)


def write_takes(folder, names, seconds=0.5):
    rows = []
    for name in names:
        for rate in (24000, 48000):
            np.save(folder / f"{name}_{rate // 1000}k.npy", np.full(int(rate * seconds), 0.1, np.float32))
        rows.append(dict(name=name, keep=True))
    rows.append(dict(name="bad_0", keep=False))
    (folder / "takes.json").write_text(json.dumps(dict(takes=rows)))


def test_takes_load_kept_ones_and_do_not_repeat_a_word(tmp_path):
    write_takes(tmp_path, ["yeah_0", "yeah_1", "mhm_0"])
    takes = Takes(tmp_path)
    assert len(takes.pcm[48000]) == 3
    rng = random.Random(0)
    assert all(takes.pick(48000, rng, "yeah")[0] == "mhm" for _ in range(20))


def test_takes_refuse_long_or_missing_audio(tmp_path):
    write_takes(tmp_path, ["yeah_0"], seconds=1.5)
    with pytest.raises(ValueError):
        Takes(tmp_path)
    (tmp_path / "takes.json").write_text(json.dumps(dict(takes=[dict(name="x_0", keep=False)])))
    with pytest.raises(ValueError):
        Takes(tmp_path)


def test_install_opts_in_per_session_and_emits_the_sound(tmp_path, monkeypatch):
    from mtplx.frankie.session import Session

    for name in ("__init__", "handle", "receive_audio"):
        monkeypatch.setattr(Session, name, getattr(Session, name))
    seen = []

    def init(self, *a, **k):
        self.listening, self.speech_start_ms, self.clock_ms = True, 0, 3000
        self.settings = {"output_rate": 48000}
        self.turn = NS(latest=reading())
        self.audible_response = lambda: None
        self.event = lambda name, **v: seen.append((name, v))

    async def handle(self, event):
        pass

    async def receive(self, *a, **k):
        pass

    monkeypatch.setattr(Session, "__init__", init)
    monkeypatch.setattr(Session, "handle", handle)
    monkeypatch.setattr(Session, "receive_audio", receive)
    write_takes(tmp_path, ["mhm_0"])
    backchannels.install(Takes(tmp_path))
    s = Session()
    asyncio.run(s.receive_audio("pcm"))
    assert seen == []  # not opted in
    with pytest.raises(ValueError):
        asyncio.run(s.handle({"type": "session.update", "session": {"frankie": {"backchannels": 1}}}))
    asyncio.run(s.handle({"type": "session.update", "session": {"frankie": {"backchannels": True}}}))
    asyncio.run(s.receive_audio("pcm"))
    asyncio.run(s.receive_audio("pcm"))  # within the gap: one sound only
    assert [name for name, _ in seen] == ["frankie.backchannel.delta"]
    event = seen[0][1]
    assert event["transcript"] == "mhm" and event["turn_ms"] == 2950
    assert len(base64.b64decode(event["delta"])) == 2 * 24000
