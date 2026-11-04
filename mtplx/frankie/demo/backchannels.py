"""Listening sounds: Frankie says "mm-hmm" while the user talks, when the turn model says a listener would.

The duplex turn model already scores, every 100 ms, how likely a backchannel is
right now (``p_bc``) and how likely Frankie is to take the turn (``p_sys``). A
sound plays when the user is mid-turn, the backchannel score is high, the turn
is not about to pass to Frankie, and none has played for a while. It never
plays over Frankie's own reply, and it never enters the conversation history.

The sounds are short takes in the session voice, rendered offline through the
live output chain (Breeze, AudioSeal, AP-BWE) and kept only if Parakeet hears
them as listening sounds: ``perf/make_backchannels.py`` writes ``takes.json``
and ``<name>_{24k,48k}.npy``. A client opts in with ``frankie.backchannels``
in ``session.update``; it then gets ``frankie.backchannel.delta`` events and
plays them in its aside slot, which any reply audio clears.
"""
import base64
import json
from pathlib import Path
import random

import numpy as np

P_BC = 0.4  # backchannel score to fire at
P_SYS = 0.15  # above this the turn may be passing to Frankie: stay quiet
MIN_TURN_MS = 1500  # into the user's turn before the first sound
GAP_MS = 6000  # between sounds
FRESH_MS = 300  # the reading must cover audio this recent


class Takes:
    def __init__(self, folder):
        folder = Path(folder)
        takes = [t for t in json.loads((folder / "takes.json").read_text())["takes"] if t["keep"]]
        if not takes:
            raise ValueError("No kept backchannel takes")
        self.pcm = {rate: [(t["name"].rsplit("_", 1)[0], np.load(folder / f"{t['name']}_{rate // 1000}k.npy"))
                           for t in takes] for rate in (24000, 48000)}
        for rate, rows in self.pcm.items():
            if any(x.dtype != np.float32 or x.ndim != 1 or not 0 < len(x) <= rate or not np.isfinite(x).all()
                   for _, x in rows):
                raise ValueError("Backchannel takes must be finite mono float32 under one second")

    def pick(self, rate, rng, last_word):
        rows = self.pcm[rate]
        fresh = [r for r in rows if r[0] != last_word] or rows
        return rng.choice(fresh)


def due(s, reading):
    """Whether the session should make a listening sound now."""
    if reading is None or not s.listening or s.audible_response() is not None:
        return False
    end_ms, p_sys, p_bc = reading[0] / 16, reading[1], reading[2]
    return (p_bc >= P_BC and p_sys < P_SYS
            and 0 <= s.clock_ms - end_ms <= FRESH_MS
            and end_ms - s.speech_start_ms >= MIN_TURN_MS
            and s.clock_ms - s.backchannel_at >= GAP_MS)


def install(takes):
    from mtplx.frankie.session import Session, identifier

    original_init, original_handle, original_receive = Session.__init__, Session.handle, Session.receive_audio

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.backchannels, self.backchannel_at, self.backchannel_word = False, -GAP_MS, None
        self.backchannel_rng = random.Random()

    async def handle(self, event):
        if event.get("type") == "session.update":
            wanted = event.get("session", {}).get("frankie", {}).get("backchannels")
            if wanted is not None and type(wanted) is not bool:
                raise ValueError("backchannels must be a boolean.")
            result = await original_handle(self, event)
            if wanted is not None:
                self.backchannels = wanted
            return result
        return await original_handle(self, event)

    async def receive_audio(self, *args, **kwargs):
        await original_receive(self, *args, **kwargs)
        reading = getattr(self.turn, "latest", None) if self.backchannels else None
        if not due(self, reading):
            return
        rate = self.settings["output_rate"]
        word, pcm = takes.pick(rate, self.backchannel_rng, self.backchannel_word)
        self.backchannel_at, self.backchannel_word = self.clock_ms, word
        delta = base64.b64encode(np.clip(pcm * 32767, -32768, 32767).astype("<i2").tobytes()).decode()
        self.event("frankie.backchannel.delta", item_id=identifier("bc"), delta=delta, transcript=word,
                   p_bc=round(reading[2], 3), p_sys=round(reading[1], 3),
                   turn_ms=round(reading[0] / 16 - self.speech_start_ms))
        print(f"Backchannel: {word} p_bc={reading[2]:.2f} p_sys={reading[1]:.2f} "
              f"{(reading[0] / 16 - self.speech_start_ms) / 1000:.1f} s into the turn", flush=True)

    Session.__init__, Session.handle, Session.receive_audio = __init__, handle, receive_audio
