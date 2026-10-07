"""Brain-led delivery: the brain's hidden states choose how Breeze says each phrase.

Every phrase reaches the mouth with the brain states that produced it. Their
mean is read against fixed directions (16 feelings, 24 words), each reading
scored against phrases of the same length, so a short "Hey there!" does not
read as the most excited thing ever said. A small adapter maps the 40
readings to weights over learned Breeze instruction row sets and to a
guidance strength (3, 3.5 or 4: every new scale costs Breeze a recompile).
The weighted rows take the instruction's place in Breeze's prompt, and paired
guidance (guidance.py) pushes toward them. No sentence is ever written.

A brain other than the one the directions were fitted on (27B) carries a
``flash-scale.npz`` that maps its readings onto the scale the adapter expects.

Assets come from one folder: ``directions.npz``, ``words.npz``,
``calib-feelings.npz``, ``calib-words.npz``, ``adapter.npz``, optionally
``flash-scale.npz``, and ``rows/NAME.slot.safetensors`` for each name in the
adapter's bank.
"""
from contextvars import ContextVar
import functools
import json
from pathlib import Path
import time

import numpy as np

PLAIN = "Speak clearly and naturally."
LEARNED = "Speak with the learned delivery."
SAMPLE_RATE = 24000


def calibrated(d, cal, n):
    """Directions d whose readings are scored against n-token phrases.

    z_cal = m(ref) + (z - m(n)) * s(ref) / s(n), with m(n) = a + b / sqrt(n) and
    s(n) = c + e / sqrt(n) per axis. Phrases at or past the reference length
    are read exactly as uncalibrated.
    """
    r0 = 1 / np.sqrt(float(cal["ref"]))
    r = max(1 / np.sqrt(max(int(n), 1)), r0)
    m, s = cal["a"] + cal["b"] * r, np.maximum(cal["c"] + cal["e"] * r, 1e-3)
    m0, s0 = cal["a"] + cal["b"] * r0, cal["c"] + cal["e"] * r0
    spread = d["spread"] * s / s0
    return {**d, "spread": spread, "center": d["center"] + d["spread"] * m - spread * m0}


def readings(d, mean):
    return ((mean - d["mu"]) @ d["V"].T - d["center"]) / d["spread"]


class Reader:
    """The brain's mean phrase state to learned rows and a guidance strength."""

    def __init__(self, folder):
        import mlx.core as mx

        folder = Path(folder)
        load = lambda name: dict(np.load(folder / name))
        self.feelings, self.words = load("directions.npz"), load("words.npz")
        self.feelings_cal, self.words_cal = load("calib-feelings.npz"), load("calib-words.npz")
        self.adapter = load("adapter.npz")
        self.scale = load("flash-scale.npz") if (folder / "flash-scale.npz").exists() else None
        self.axes = [str(x) for x in self.feelings["axes"]]
        self.bank = [str(x) for x in self.adapter["bank"]]
        self.rows = {n: mx.load(str(folder / "rows" / f"{n}.slot.safetensors"))["theta"] for n in self.bank}
        self.width = int(self.feelings["mu"].shape[-1])
        if self.words["mu"].shape[-1] != self.width:
            raise ValueError("Feeling and word directions read different brain widths")

    def read(self, mean, n):
        """mean: the phrase's brain states averaged over its n rows."""
        zf = readings(calibrated(self.feelings, self.feelings_cal, n), mean)
        zw = readings(calibrated(self.words, self.words_cal, n), mean)
        if self.scale is not None:
            s = self.scale
            zf = s["fa"] + (zf - s["fc"]) / s["fe"] * s["fb"]
            zw = s["wa"] + (zw - s["wc"]) / s["we"] * s["wb"]
        a = self.adapter
        f = np.r_[(np.r_[zf, zw] - a["mu"]) / a["sd"], 1.0]
        logits = f @ a["W"]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        kept = {self.bank[j]: float(p[j]) for j in np.argsort(-p) if p[j] >= 0.05}
        total = sum(kept.values())
        weights = {name: round(x / total, 3) for name, x in kept.items()}
        return dict(weights=weights,
                    rows=sum(x * self.rows[name] for name, x in weights.items()),
                    strength=float(np.round(np.clip(f @ a["A"], 3, 4) * 2) / 2),
                    feelings=np.array([round(float(x), 2) for x in zf]))


class Delivery:
    """Wraps one BreezeMouth so the brain leads every phrase it is given states for."""

    def __init__(self, mouth, reader, *, penalty=1.1, log=None):
        from mlx_audio.tts.models.breeze_tts import Model

        from .guidance import PairedGuidance, build_paired_depth

        self.mouth, self.reader, self.log = mouth, reader, Path(log) if log else None
        self.phrases = 0
        model = mouth.model
        generate = model.generate
        model.generate = functools.wraps(generate)(
            lambda *a, **k: generate(*a, **k, repetition_penalty=penalty))
        # Guided phrases run the upstream loop; guidance does BreezeModel's accounting for both lanes.
        upstream = functools.partial(Model.generate, model)
        self.guidance = None
        self._make_guidance = lambda: PairedGuidance(
            model, lambda *a, **k: upstream(*a, repetition_penalty=penalty, **k),
            build_paired_depth(model)).install()
        self._speak = mouth.speak
        self._inside = ContextVar("brain_led_inside", default=False)
        mouth.speak = self.speak

    def speak(self, text, states, **kwargs):
        if self._inside.get() or kwargs.get("speech_instruction") is not None:
            # BreezeMouth calling itself, or an instruction the client chose.
            yield from self._speak(text, states, **kwargs)
            return
        if self.guidance is None:
            # Built at the first phrase, after the server has finished wrapping generate.
            self.guidance = self._make_guidance()
        g, model = self.guidance, self.mouth.model
        row = None
        if getattr(states, "ndim", 0) == 2 and len(states) and states.shape[1] == self.reader.width:
            import mlx.core as mx
            row = self.reader.read(np.asarray(mx.mean(states.astype(mx.float32), axis=0)), len(states))
        led = g.in_step()
        if led:
            model.instruction_rows = row and row["rows"]
            g.scale = max(1.001, row["strength"] if row else 1.0)
        instruction = LEARNED if led and row else PLAIN
        started, first, samples = time.perf_counter(), None, 0
        token = self._inside.set(True)
        try:
            for pcm in self._speak(text, states, **{**kwargs, "speech_instruction": instruction}):
                if first is None:
                    first = time.perf_counter() - started
                samples += len(pcm)
                yield pcm
        finally:
            self._inside.reset(token)
            g.scale, model.instruction_rows = 1.0, None
            self._record(text, states, row, led, first, samples, started)

    def _record(self, text, states, row, led, first, samples, started):
        """With a log folder: each phrase's choice in delivery.jsonl, its states in states/N.npy."""
        if self.log is None:
            return
        import mlx.core as mx

        n, self.phrases = self.phrases, self.phrases + 1
        folder = self.log / "states"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{n}.npy", np.asarray(mx.array(states).astype(mx.float32)).astype(np.float16))
        entry = dict(n=n, text=text, led=led, rows=len(states),
                     first_ms=first and round(first * 1000, 1), speech_s=round(samples / SAMPLE_RATE, 2),
                     took_s=round(time.perf_counter() - started, 2))
        if row is not None:
            entry.update(weights=row["weights"], strength=row["strength"], feelings=row["feelings"].tolist())
        with open(self.log / "delivery.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


def install(engine_class, folder, *, penalty=1.1, log=None):
    """Every Frankie built from engine_class reads its brain states and leads its mouth."""
    from .features import FinalFeatures

    reader = Reader(folder)
    original = engine_class.__init__

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.audio.breeze is None:
            raise ValueError("Brain-led delivery needs the Breeze mouth")
        self._expression_feature_stream = functools.partial(FinalFeatures, width=reader.width)
        self._delivery = Delivery(self.audio.breeze, reader, penalty=penalty, log=log)

    engine_class.__init__ = __init__
    return reader
