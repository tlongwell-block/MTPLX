"""Causal AudioSeal streams for 24 kHz speech, CPU eager only.

``WatermarkPool`` shares one streaming generator across responses with
serialized access; each ``Response`` owns its encoder and decoder state.
``ResidualResponse`` runs that 16 kHz generator on 24 kHz speech without
resampling the speech itself: it resamples down, watermarks, and adds back
only the watermark residual, resampled up through causal rational FIRs. The
two FIRs each have 30 high-rate samples of group delay at 48 kHz; their
combined 30 / 24000 s = 1.25 ms is compensated by holding back original PCM,
so the original stays bit exact in float until the addition.
"""
import copy
import threading

import numpy as np
from scipy.signal import firwin, upfirdn
import torch
import torch.nn.functional as F


def compact_state(value):
    """Copy state tensors so they do not retain large intermediate views."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: compact_state(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(compact_state(v) for v in value)
    if isinstance(value, list):
        return [compact_state(v) for v in value]
    if hasattr(value, "__dict__"):
        out = copy.copy(value)
        for k, v in vars(value).items():
            setattr(out, k, compact_state(v))
        return out
    return value


class WatermarkPool:
    def __init__(self, model, compact=False):
        self.model = model
        self.compact = compact
        self.lock = threading.RLock()

    def response(self, payload):
        return Response(self, payload)


class Response:
    def __init__(self, pool, payload):
        self.pool = pool
        self._payload = torch.as_tensor(payload, dtype=torch.int64).reshape(1, -1).clone()
        if self._payload.shape != (1, 16) or not torch.all((self._payload == 0) | (self._payload == 1)):
            raise ValueError("Expected fixed 16 generic bits")
        self.state = None
        self.pending = torch.empty(1, 1, 0)
        self.cancelled = threading.Event()
        self.closed = False
        self.input_samples = 0
        self.output_samples = 0

    def cancel(self):
        # Set before waiting for the lock, so an active worker discards its output.
        self.cancelled.set()
        with self.pool.lock:
            self.pending = self.pending[..., :0]
            self.state = None
            self.closed = True

    def _infer(self, x):
        m = self.pool.model
        with torch.inference_mode(), m.streaming(batch_size=1):
            if self.state is not None:
                m.set_streaming_state(self.state)
            y = m(x, message=self._payload)
            self.state = m.get_streaming_state()
            if self.pool.compact:
                self.state = compact_state(self.state)
        return y

    def push(self, x):
        with self.pool.lock:
            if self.cancelled.is_set():
                return torch.empty(1, 1, 0)
            if self.closed:
                raise RuntimeError("Response finished")
            x = torch.as_tensor(x, dtype=torch.float32).reshape(1, 1, -1)
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite PCM")
            self.input_samples += x.shape[-1]
            joined = torch.cat((self.pending, x), dim=-1)
            n = joined.shape[-1] // self.pool.model.frame_size * self.pool.model.frame_size
            self.pending = joined[..., n:].clone()
            if n == 0:
                return joined[..., :0]
            y = self._infer(joined[..., :n])
            if self.cancelled.is_set():
                return y[..., :0]
            self.output_samples += y.shape[-1]
            return y

    def finish(self):
        with self.pool.lock:
            if self.closed or self.cancelled.is_set():
                return torch.empty(1, 1, 0)
            n = self.pending.shape[-1]
            y = self.pending
            if n:
                y = self._infer(F.pad(self.pending, (0, self.pool.model.frame_size - n)))[..., :n]
            if self.cancelled.is_set():
                y = y[..., :0]
            self.pending = self.pending[..., :0]
            self.state = None
            self.closed = True
            self.output_samples += y.shape[-1]
            return y


class CausalResampler:
    def __init__(self, up, down):
        self.up = up
        self.down = down
        self.h = (firwin(61, 1 / max(up, down), window=("kaiser", 5.0)) * up).astype(np.float64)
        self.buf = np.empty(0)
        self.start = 0
        self.nin = 0
        self.nout = 0

    def push(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not len(x):
            return x
        self.buf = np.concatenate((self.buf, x))
        self.nin += len(x)
        target = (self.nin * self.up + self.down - 1) // self.down
        raw = upfirdn(self.h, self.buf, up=self.up, down=self.down)
        offset = self.start * self.up // self.down
        y = raw[self.nout - offset:target - offset].copy()
        self.nout = target
        newstart = max(0, ((self.nin - len(self.h)) // self.down) * self.down)
        self.buf = self.buf[newstart - self.start:]
        self.start = newstart
        return y


class ResidualResponse:
    def __init__(self, pool, payload):
        self.response = pool.response(payload)
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self.down = CausalResampler(2, 3)
        self.up = CausalResampler(3, 2)
        self.original = np.empty(0, dtype=np.float32)
        self.low_original = np.empty(0, dtype=np.float32)
        self.skip = 30
        self.closed = False
        self.input_samples = 0
        self.output_samples = 0

    def _low(self, x):
        low = self.down.push(x).astype(np.float32)
        self.low_original = np.concatenate((self.low_original, low))
        marked = self.response.push(torch.from_numpy(low)).numpy().reshape(-1)
        n = len(marked)
        res = marked - self.low_original[:n]
        self.low_original = self.low_original[n:]
        return self.up.push(res)

    def _emit(self, res):
        if self._cancelled.is_set():
            return torch.empty(1, 1, 0)
        n = min(self.skip, len(res))
        res = res[n:]
        self.skip -= n
        n = min(len(res), len(self.original))
        out = self.original[:n] + res[:n].astype(np.float32)
        self.original = self.original[n:]
        self.output_samples += n
        return torch.from_numpy(out).reshape(1, 1, -1)

    def push(self, x):
        with self._lock:
            if self.closed or self._cancelled.is_set():
                return torch.empty(1, 1, 0)
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            self.original = np.concatenate((self.original, x))
            self.input_samples += len(x)
            return self._emit(self._low(x))

    def finish(self):
        with self._lock:
            if self.closed or self._cancelled.is_set():
                return torch.empty(1, 1, 0)
            # Advance the causal filters past the final source sample. This is
            # normal completion only; output never exceeds the original count.
            outs = [self._emit(self._low(np.zeros(96, dtype=np.float32)))]
            tail = self.response.finish().numpy().reshape(-1)
            res = tail - self.low_original[:len(tail)]
            self.low_original = self.low_original[len(tail):]
            outs.append(self._emit(self.up.push(res)))
            outs.append(self._emit(self.up.push(np.zeros(64))))
            if len(self.original):
                raise AssertionError("Residual flush lost samples")
            self.closed = True
            return torch.cat(outs, -1)

    def cancel(self):
        # Signal before either lock so in-flight work cannot emit a completion tail.
        self._cancelled.set()
        self.response.cancel()
        with self._lock:
            self.original = np.empty(0, dtype=np.float32)
            self.low_original = np.empty(0, dtype=np.float32)
            self.closed = True
