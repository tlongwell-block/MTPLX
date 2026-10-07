"""Silero v5 VAD on the CPU; reused from Frankie's MLX reference.

All inference operations specify the CPU stream so listening cannot move the
brain or mouth onto the CPU or wait behind their GPU work.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn


class SileroVAD(nn.Module):
    N_FFT, STRIDE, CTX, HOP = (256, 128, 64, 512)

    def __init__(self):
        super().__init__()
        self.stft_conv = nn.Conv1d(1, 258, 256, stride=128, bias=False)
        self.conv1 = nn.Conv1d(129, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1)
        self.lstm_cell = nn.LSTM(128, 128)
        self.final_conv = nn.Conv1d(128, 1, 1)
        self.reset()

    def reset(self):
        # These are lazy: a GPU reset would make the next CPU audio frame
        # wait behind unrelated brain/mouth work on the default stream.
        self._h = mx.zeros((1, 128), stream=mx.cpu)
        self._c = mx.zeros((1, 128), stream=mx.cpu)
        self._ctx = mx.zeros((1, self.CTX), stream=mx.cpu)

    def prepare(self):
        self._WxT, self._WhT, self._fwT = (
            self.lstm_cell.Wx.T,
            self.lstm_cell.Wh.T,
            self.final_conv.weight[0].T,
        )
        mx.eval(
            self.parameters(),
            self._WxT,
            self._WhT,
            self._fwT,
            self._h,
            self._c,
            self._ctx,
        )
        return self

    def features(self, x, s):
        x = mx.pad(x, [(0, 0), (0, self.N_FFT // 4)], mode="reflect", stream=s)
        x = mx.expand_dims(x, 2, stream=s)
        z = mx.conv1d(x, self.stft_conv.weight, stride=self.STRIDE, stream=s)
        re, im = mx.split(z, 2, axis=-1, stream=s)
        z = mx.sqrt(
            mx.add(mx.square(re, stream=s), mx.square(im, stream=s), stream=s), stream=s
        )
        for c, st in (
            (self.conv1, 1),
            (self.conv2, 2),
            (self.conv3, 2),
            (self.conv4, 1),
        ):
            z = mx.maximum(
                mx.add(
                    mx.conv1d(z, c.weight, stride=st, padding=1, stream=s),
                    c.bias,
                    stream=s,
                ),
                0,
                stream=s,
            )
        return mx.squeeze(z, 1, stream=s)

    def __call__(self, chunk):
        s = mx.cpu
        x = mx.expand_dims(
            mx.array(np.ascontiguousarray(chunk, dtype=np.float32)), 0, stream=s
        )
        assert x.shape[1] == self.HOP, x.shape
        f = self.features(mx.concatenate([self._ctx, x], axis=1, stream=s), s)
        self._ctx = mx.slice(
            x, mx.array([self.HOP - self.CTX]), (1,), (1, self.CTX), stream=s
        )
        g = mx.addmm(self.lstm_cell.bias, f, self._WxT, stream=s)
        g = mx.addmm(g, self._h, self._WhT, stream=s)
        i, fg, gg, o = mx.split(g, 4, axis=-1, stream=s)
        c = mx.add(
            mx.multiply(mx.sigmoid(fg, stream=s), self._c, stream=s),
            mx.multiply(mx.sigmoid(i, stream=s), mx.tanh(gg, stream=s), stream=s),
            stream=s,
        )
        h = mx.multiply(mx.sigmoid(o, stream=s), mx.tanh(c, stream=s), stream=s)
        self._h, self._c = (h, c)
        p = mx.sigmoid(
            mx.addmm(
                self.final_conv.bias, mx.maximum(h, 0, stream=s), self._fwT, stream=s
            ),
            stream=s,
        )
        mx.eval(p, self._h, self._c, self._ctx)
        return float(p.item()) if p.size == 1 else float(p[0, 0])
