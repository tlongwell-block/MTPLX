"""Streaming MaAI CPC/ALiBi turn projection on CPU, inside the server process.

Port of the shared mtmd turn graph; MaAI is Copyright (c) 2025 MaAI
Development Team, MIT licensed. Inputs are aligned 100 ms stereo frames.
"""

import queue
import threading

import mlx.core as mx
import numpy as np


class TurnModel:
    def __init__(self, weights, mode="vap"):
        if mode not in {"vap", "bc", "duplex"}:
            raise ValueError("Unsupported turn projection mode.")
        self.w = weights
        self.mode = mode
        self.reset()

    def reset(self):
        self.previous = [np.zeros(320, np.float32) for _ in range(2)]
        self.recurrent = [
            (mx.zeros((256,), stream=mx.cpu), mx.zeros((256,), stream=mx.cpu))
            for _ in range(2)
        ]
        self.cache = {}

    def linear(self, x, name):
        return mx.matmul(
            x, mx.transpose(self.w[name + ".weight"], stream=mx.cpu), stream=mx.cpu
        )

    def norm(self, x, name):
        w = self.w
        return mx.fast.layer_norm(
            x, w[name + ".weight"], w[name + ".bias"], 1e-5, stream=mx.cpu
        )

    def encoder(self, pcm, channel):
        s, w = mx.cpu, self.w
        x = mx.array(np.concatenate([self.previous[channel], pcm]))[None, :, None]
        self.previous[channel] = pcm[-320:].copy()
        for i, (stride, pad) in enumerate(zip((5, 4, 2, 2, 2), (3, 2, 1, 1, 1))):
            name = f"encoder.encoder.gEncoder.conv{i}"
            x = mx.conv1d(x, w[name + ".weight"], stride=stride, padding=pad, stream=s)
            x = mx.add(x, w[name + ".bias"], stream=s)
            centered = mx.subtract(
                x, mx.mean(x, axis=-1, keepdims=True, stream=s), stream=s
            )
            variance = mx.var(x, axis=-1, keepdims=True, ddof=1, stream=s)
            x = mx.divide(
                centered, mx.sqrt(mx.add(variance, 1e-5, stream=s), stream=s), stream=s
            )
            name = f"encoder.encoder.gEncoder.batchNorm{i}"
            x = mx.maximum(
                mx.add(
                    mx.multiply(x, w[name + ".weight"], stream=s),
                    w[name + ".bias"],
                    stream=s,
                ),
                0,
                stream=s,
            )
        x = mx.squeeze(
            mx.slice(x, mx.array([1]), (1,), (1, 10, 256), stream=s), 0, stream=s
        )
        name = "encoder.encoder.gAR.baseNet."
        wx = mx.addmm(
            mx.add(w[name + "bias_ih_l0"], w[name + "bias_hh_l0"], stream=s),
            x,
            mx.transpose(w[name + "weight_ih_l0"], stream=s),
            stream=s,
        )
        h, c = self.recurrent[channel]
        rows = []
        for t in range(10):
            row = mx.squeeze(mx.take(wx, mx.array([t]), axis=0, stream=s), 0, stream=s)
            gates = mx.addmm(
                row, h, mx.transpose(w[name + "weight_hh_l0"], stream=s), stream=s
            )
            i, f, g, o = mx.split(gates, 4, axis=-1, stream=s)
            c = mx.add(
                mx.multiply(mx.sigmoid(f, stream=s), c, stream=s),
                mx.multiply(mx.sigmoid(i, stream=s), mx.tanh(g, stream=s), stream=s),
                stream=s,
            )
            h = mx.multiply(mx.sigmoid(o, stream=s), mx.tanh(c, stream=s), stream=s)
            rows.append(h)
        self.recurrent[channel] = h, c
        x = mx.expand_dims(mx.stack(rows, stream=s), 0, stream=s)
        x = mx.conv1d(x, w["encoder.downsample.1.weight"], stride=2, stream=s)
        x = mx.add(x, w["encoder.downsample.1.bias"], stream=s)
        return self.gelu(
            self.norm(mx.squeeze(x, 0, stream=s), "encoder.downsample.2.ln")
        )

    @staticmethod
    def gelu(x):
        s = mx.cpu
        return mx.multiply(
            mx.multiply(x, 0.5, stream=s),
            mx.add(1, mx.erf(mx.multiply(x, 2**-0.5, stream=s), stream=s), stream=s),
            stream=s,
        )

    def attention(self, query, source, name, channel):
        s = mx.cpu
        q, k, v = (
            self.linear(x, name + "." + part)
            for x, part in ((query, "query"), (source, "key"), (source, "value"))
        )
        key = (name, channel)
        past = self.cache.get(key)
        if past is not None:
            k, v = (
                mx.concatenate([old, new], axis=0, stream=s)
                for old, new in zip(past, (k, v))
            )
        n, length = q.shape[0], k.shape[0]
        self.cache[key] = tuple(
            mx.slice(
                x,
                mx.array([max(0, length - 199)]),
                (0,),
                (min(199, length), 256),
                stream=s,
            )
            for x in (k, v)
        )
        q, k, v = (
            mx.transpose(mx.reshape(x, (-1, 4, 64), stream=s), (1, 0, 2), stream=s)
            for x in (q, k, v)
        )
        scores = mx.multiply(
            mx.matmul(q, mx.swapaxes(k, -1, -2, stream=s), stream=s), 1 / 16, stream=s
        )
        positions = mx.arange(length, stream=s)
        bias = mx.multiply(
            mx.reshape(self.w[name + ".m"], (4, 1, 1), stream=s),
            mx.reshape(positions, (1, 1, length), stream=s),
            stream=s,
        )
        query_positions = (length - n + np.arange(n))[:, None]
        positions = np.arange(length)[None, :]
        mask = (positions <= query_positions) & (positions > query_positions - 200)
        scores = mx.where(
            mx.array(mask)[None],
            mx.add(scores, bias, stream=s),
            -float("inf"),
            stream=s,
        )
        x = mx.matmul(mx.softmax(scores, axis=-1, stream=s), v, stream=s)
        x = mx.reshape(mx.transpose(x, (1, 0, 2), stream=s), (n, 256), stream=s)
        return self.linear(x, name + ".proj")

    def layer(self, x, other, name, channel):
        s = mx.cpu
        z = self.norm(x, name + ".ln_self_attn")
        x = mx.add(x, self.attention(z, z, name + ".mha", channel), stream=s)
        if other is not None:
            x = mx.add(
                x,
                self.attention(
                    self.norm(x, name + ".ln_src_attn"),
                    other,
                    name + ".mha_cross",
                    channel,
                ),
                stream=s,
            )
        z = self.gelu(
            self.linear(self.norm(x, name + ".ln_ffnetwork"), name + ".ffnetwork.0")
        )
        return mx.add(x, self.linear(z, name + ".ffnetwork.3"), stream=s)

    def process(self, user, system):
        s = mx.cpu
        if (
            user.shape != (1600,)
            or system.shape != (1600,)
            or not np.isfinite([user, system]).all()
        ):
            raise ValueError(
                "Turn prediction requires finite aligned 100 ms frames at 16 kHz."
            )
        audio = (user, system) if self.mode == "vap" else (system, user)
        a, b = (
            self.layer(self.encoder(x, i), None, "ar_channel.layers.0", i)
            for i, x in enumerate(audio)
        )
        for i in range(3):
            name = f"ar.layers.{i}"
            a, b = self.layer(a, b, name, 0), self.layer(b, a, name, 1)
        x = mx.add(
            self.gelu(
                self.norm(self.linear(a, "ar.combinator.h0_a"), "ar.combinator.ln")
            ),
            self.gelu(
                self.norm(self.linear(b, "ar.combinator.h0_b"), "ar.combinator.ln")
            ),
            stream=s,
        )
        outputs = []
        if self.mode != "bc":
            logits = mx.add(
                self.linear(x, "vap_head"), self.w["vap_head.bias"], stream=s
            )
            now = self.linear(mx.softmax(logits, axis=-1, stream=s), "now")
            outputs.append(
                mx.divide(
                    now,
                    mx.add(
                        mx.sum(now, axis=-1, keepdims=True, stream=s), 1e-5, stream=s
                    ),
                    stream=s,
                )
            )
        if self.mode != "vap":
            logits = mx.add(self.linear(x, "bc_head"), self.w["bc_head.bias"], stream=s)
            outputs.append(mx.sigmoid(logits, stream=s))
        mx.eval(outputs, self.recurrent, self.cache)
        values = [np.asarray(output)[-1].copy() for output in outputs]
        if self.mode == "duplex":
            # BC encodes (system, user); expose (user, system, backchannel).
            values[0] = values[0][::-1]
        return np.concatenate(values)


class TurnWorker:
    def __init__(self, weights, mode="vap"):
        self.model = TurnModel(weights, mode)
        self.queue = queue.Queue(maxsize=30)
        self.latest = None
        self.failed = False
        self.pending = np.empty((2, 0), np.float32)
        self.samples = 0
        self.epoch = 0
        self.needs_reset = False
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="frankie-turn"
        )
        self.thread.start()

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            end, audio, reset, epoch = item
            try:
                if reset:
                    self.model.reset()
                probability = self.model.process(*audio)
                if epoch == self.epoch:
                    self.latest = (
                        end,
                        float(probability[1]),
                        float(probability[2]) if len(probability) == 3 else 0.0,
                    )
            except Exception as error:  # noqa: BLE001 — stop a failed inference worker cleanly.
                self.failed = True
                self.latest = None
                print(f"Turn prediction disabled: {error}", flush=True)
                return

    def append(self, user, system):
        if self.failed:
            return
        self.pending = np.concatenate([self.pending, np.stack([user, system])], axis=1)
        while self.pending.shape[1] >= 1600:
            audio, self.pending = self.pending[:, :1600].copy(), self.pending[:, 1600:]
            self.samples += 1600
            reset = self.needs_reset
            self.needs_reset = False
            if self.queue.full():
                while True:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
                self.latest = None
                reset = True
            self.queue.put_nowait((self.samples, audio, reset, self.epoch))

    def reset(self, clock_ms):
        self.epoch += 1
        self.latest = None
        self.samples = round(clock_ms * 16)
        self.pending = np.empty((2, 0), np.float32)
        self.needs_reset = True
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def release(self, clock_ms, silence_ms):
        reading = self.latest
        return (
            self.failed
            or reading is None
            or silence_ms >= 1500
            or not 0 <= clock_ms - reading[0] / 16 <= 500
            or reading[1] >= 0.4
        )

    def close(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put_nowait(None)
        self.thread.join(timeout=2)
