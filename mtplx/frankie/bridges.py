"""Frankie trained ear adapters, extracted from its reference MLX implementation."""

import mlx.core as mx
from mlx import nn


class EarBridge(nn.Module):
    """== v3_ear_train2.Adapter: softmax(CTC(x)) @ M (frozen computed seam) + gate * MLP(LN(x)). Weights from the G1/G5 checkpoint."""

    def __init__(self, n_units, d_brain, hidden=1024):
        super().__init__()
        self.ctc = nn.Linear(512, n_units)
        self.proj = nn.Linear(n_units, d_brain, bias=False)
        self.ln = nn.LayerNorm(512)
        self.l1 = nn.Linear(512, hidden)
        self.l2 = nn.Linear(hidden, d_brain)
        self.gate = mx.zeros((1,))

    def __call__(self, x, *, ctc=None):
        ctc = self.ctc(x) if ctc is None else ctc
        return self.proj(mx.softmax(ctc, axis=-1)) + self.gate * self.l2(
            nn.gelu(self.l1(self.ln(x)))
        )


class EarTone(nn.Module):
    """== g5b_tone.ToneToken: [mean, std] over ear frames -> frozen logistic probe -> probs @ E(angry..sad) + gate * MLP(LN(pool)). One row."""

    def __init__(self, word_emb, hidden=1024):
        super().__init__()
        self.probe = nn.Linear(1024, 6)
        self.ln = nn.LayerNorm(1024)
        self.l1 = nn.Linear(1024, hidden)
        self.l2 = nn.Linear(hidden, word_emb.shape[1])
        self.gate = mx.zeros((1,))
        self._W = word_emb

    def pool(self, x):
        return mx.concatenate([x.mean(0), mx.sqrt(x.var(0) + 1e-06)])

    def probs(self, x):
        return mx.softmax(self.probe(self.pool(x)), axis=-1)

    def __call__(self, x):
        p = self.pool(x)
        return (
            mx.softmax(self.probe(p), axis=-1) @ self._W
            + self.gate * self.l2(nn.gelu(self.l1(self.ln(p))))
        )[None]
