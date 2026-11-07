"""48 kHz speech: Frankie's 24 kHz PCM gains the octave above 12 kHz.

Breeze's codec stops at 12 kHz, the top of a 24 kHz signal. The breath and air
above it are much of what makes a voice sound present. AP-BWE (Lu et al.
2024, MIT; weights ``g_24kto48k``) predicts the missing amplitude and phase
spectra of the 2x-upsampled signal. Only those high bins are taken: below the
crossover every STFT bin is the input's own, so the speech, and the watermark,
pass through as they were.

``install`` wraps ``Frankie.respond`` for responses whose ``output_rate`` is
48000; 24 kHz responses are untouched. ``Stream`` is causal with a fixed
lookahead: each push emits what it can see LOOKAHEAD past, the rest waits for
the next push or the response's end. Windows start on the STFT hop grid, so
any partition of the input renders the same samples as the whole signal. A
phrase mark waits until its last sample is out: flushing there would render
its tail without the speech that follows.
"""
from collections import deque
from functools import wraps
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

RATE, OUTPUT_RATE = 24000, 48000
CROSSOVER_HZ = (10500, 11500)  # input bins below, a linear blend across, model bins above
LOOKAHEAD = 960  # 40 ms at 24 kHz: streamed output matches a whole-signal render (-95 dB above 12 kHz)
CONTEXT = 2880  # 120 ms of history per window
HOP = 80  # window origins that reproduce the whole render (measured: 40, one STFT hop, does not)


def _upsampler(width=6, rolloff=0.99):
    """torchaudio's default sinc-Hann 2x kernel, the front end AP-BWE was trained on.
    Its gentle band edge leaves the 12 kHz shoulder the model extends from."""
    base, reach = rolloff, int(np.ceil(width / rolloff))
    t = (np.arange(0, -2, -1)[:, None] / 2 + np.arange(-reach, reach + 1)[None]) * base
    t = np.clip(t, -width, width)
    window = np.cos(t * np.pi / width / 2) ** 2
    t = t * np.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.where(t == 0, 1.0, np.sin(t) / t)
    return (kernel * window * base).astype(np.float32), reach


UPSAMPLE, REACH = _upsampler()


def upsample(pcm24):
    x = np.pad(np.asarray(pcm24, np.float32), (REACH, REACH + 1))
    y = np.empty((2, len(pcm24)), np.float32)
    for phase in range(2):
        y[phase] = np.correlate(x, UPSAMPLE[phase], mode="valid")[:len(pcm24)]
    return y.T.reshape(-1)


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * 3)
        self.pwconv2 = nn.Linear(dim * 3, dim)
        self.gamma = mx.ones((dim,))

    def __call__(self, x):
        return x + self.gamma * self.pwconv2(nn.gelu(self.pwconv1(self.norm(self.dwconv(x)))))


class Generator(nn.Module):
    """AP-BWE: interacting ConvNeXt stacks over log amplitude and phase."""

    def __init__(self, n_fft, dim, layers):
        super().__init__()
        bins = n_fft // 2 + 1
        self.conv_pre_mag = nn.Conv1d(bins, dim, 7, padding=3)
        self.conv_pre_pha = nn.Conv1d(bins, dim, 7, padding=3)
        self.norm_pre_mag = nn.LayerNorm(dim, eps=1e-6)
        self.norm_pre_pha = nn.LayerNorm(dim, eps=1e-6)
        self.convnext_mag = [Block(dim) for _ in range(layers)]
        self.convnext_pha = [Block(dim) for _ in range(layers)]
        self.norm_post_mag = nn.LayerNorm(dim, eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(dim, eps=1e-6)
        self.linear_post_mag = nn.Linear(dim, bins)
        self.linear_post_pha_r = nn.Linear(dim, bins)
        self.linear_post_pha_i = nn.Linear(dim, bins)

    def __call__(self, mag, pha):  # (1, frames, bins)
        xm = self.norm_pre_mag(self.conv_pre_mag(mag))
        xp = self.norm_pre_pha(self.conv_pre_pha(pha))
        for bm, bp in zip(self.convnext_mag, self.convnext_pha):
            xm = xm + xp
            xp = xp + xm
            xm, xp = bm(xm), bp(xp)
        mag = mag + self.linear_post_mag(self.norm_post_mag(xm))
        xp = self.norm_post_pha(xp)
        return mag, mx.arctan2(self.linear_post_pha_i(xp), self.linear_post_pha_r(xp))


class Extender:
    """Window in, 2x window out. Thread-agnostic: weights are evaluated at load."""

    def __init__(self, folder):
        folder = Path(folder)
        h = json.loads((folder / "config.json").read_text())
        if (h["lr_sampling_rate"], h["hr_sampling_rate"]) != (RATE, OUTPUT_RATE):
            raise ValueError("Expected AP-BWE 24 kHz -> 48 kHz weights")
        self.n_fft, self.hop, win = h["n_fft"], h["hop_size"], h["win_size"]
        self.model = Generator(self.n_fft, h["ConvNeXt_channels"], h["ConvNeXt_layers"])
        weights = mx.load(str(folder / "g_24kto48k.safetensors"))
        # torch Conv1d (out, in, k) -> MLX (out, k, in)
        self.model.load_weights([(k, v.transpose(0, 2, 1) if v.ndim == 3 else v)
                                 for k, v in weights.items()])
        mx.eval(self.model.parameters())
        window = np.zeros(self.n_fft, np.float32)
        pad = (self.n_fft - win) // 2
        window[pad:pad + win] = np.hanning(win + 1)[:-1]  # periodic, as torch.hann_window
        self.window = window
        bins = np.fft.rfftfreq(self.n_fft, 1 / OUTPUT_RATE)
        lo, hi = CROSSOVER_HZ
        self.blend = mx.array(np.clip((bins - lo) / (hi - lo), 0, 1).astype(np.float32))

    def __call__(self, pcm24):
        x = upsample(pcm24)
        n, half = len(x), self.n_fft // 2
        padded = np.pad(x, half, mode="reflect")
        frames = 1 + (len(padded) - self.n_fft) // self.hop
        index = np.arange(self.n_fft)[None] + self.hop * np.arange(frames)[:, None]
        spec = mx.fft.rfft(mx.array(padded[index] * self.window), axis=-1)
        mag, pha = mx.log(mx.abs(spec) + 1e-4), mx.arctan2(spec.imag, spec.real)
        mag2, pha2 = self.model(mag[None], pha[None])
        amp = mx.exp(mag2[0])
        wide = amp * mx.cos(pha2[0]) + 1j * (amp * mx.sin(pha2[0]))
        spec = spec + self.blend * (wide - spec)
        out = np.array(mx.fft.irfft(spec, n=self.n_fft, axis=-1)) * self.window
        total = self.n_fft + self.hop * (frames - 1)
        y, norm, w2 = np.zeros(total, np.float32), np.zeros(total, np.float32), self.window ** 2
        for t in range(frames):
            s = t * self.hop
            y[s:s + self.n_fft] += out[t]
            norm[s:s + self.n_fft] += w2
        return (y / np.maximum(norm, 1e-11))[half:half + n]


class Stream:
    """One response's causal 24 -> 48 kHz stream."""

    def __init__(self, extender, *, lookahead=LOOKAHEAD, context=CONTEXT, hop=HOP):
        self.extend, self.lookahead, self.context, self.hop = extender, lookahead, context, hop
        self.pcm = np.empty(0, np.float32)  # input from absolute sample `start`
        self.start = self.done = self.received = 0

    def push(self, pcm):
        self.pcm = np.concatenate((self.pcm, np.asarray(pcm, np.float32).reshape(-1)))
        self.received += len(pcm)
        return self._emit(self.received - self.lookahead)

    def finish(self):
        return self._emit(self.received)

    def cancel(self):
        self.pcm = self.pcm[:0]

    def _emit(self, end):
        if end <= self.done:
            return np.empty(0, np.float32)
        lo = self._origin(self.done)
        hi = min(self.received, end + self.lookahead)
        y = self.extend(self.pcm[lo - self.start:hi - self.start])
        out = y[2 * (self.done - lo):2 * (end - lo)]
        self.done = end
        keep = self._origin(self.done)
        self.pcm, self.start = self.pcm[keep - self.start:], keep
        return out

    def _origin(self, done):
        # Context before `done`, rounded down onto the hop grid (start is on it too).
        return max(self.start, (done - self.context) // self.hop * self.hop)


def install(engine_class, extender, report=None):
    original = engine_class.respond

    @wraps(original)
    def respond(self, items, settings, emit, abort, *, session_id):
        if ("audio" not in settings["output_modalities"]
                or settings.get("output_rate", RATE) != OUTPUT_RATE):
            return original(self, items, settings, emit, abort, session_id=session_id)
        stream, marks = Stream(extender), deque()

        def publish(pcm):
            if not np.isfinite(pcm).all():
                raise ValueError("Nonfinite 48 kHz PCM")
            if abort.is_set():
                raise InterruptedError("Response cancelled during 48 kHz extension")
            if len(pcm):
                emit("audio", pcm)
            while marks and stream.done >= marks[0][0]:
                emit("chunk", marks.popleft()[1])

        def filtered(kind, value):
            if abort.is_set():
                raise InterruptedError("Response cancelled before 48 kHz extension")
            if kind == "audio":
                publish(stream.push(value))
            elif kind == "chunk":
                marks.append((stream.received, value))
                publish(np.empty(0, np.float32))
            else:
                emit(kind, value)

        try:
            result = original(self, items, settings, filtered, abort, session_id=session_id)
            if abort.is_set():
                raise InterruptedError("Response cancelled before 48 kHz flush")
            publish(stream.finish())
            if marks or stream.done != stream.received:
                raise ValueError("48 kHz extension did not conserve response samples")
            return result
        finally:
            stream.cancel()

    engine_class.respond = respond
    engine_class.output_rates = (RATE, OUTPUT_RATE)
