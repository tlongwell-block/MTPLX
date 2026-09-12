"""Executed-token features for consumers of the ordinary generation callbacks.

Speculative windows may contain rejected tokens and the last emitted token may
not have run yet. Keep observation separate from commitment: only matching rows
below the live cache boundary can leave this stream.
"""

from __future__ import annotations


class CommittedFeatures:
    def __init__(self, runtime, prompt_length, callback, *, layer=16):
        self.model = getattr(runtime.model, "language_model", runtime.model)
        self.layer = layer
        self.prompt_length = prompt_length
        self.callback = callback
        self.tokens = []
        self.emitted = 0
        self.rows = {}
        self.cache = None

    @staticmethod
    def offset(cache):
        return next((int(c.offset) for c in cache or [] if hasattr(c, "offset")), 0)

    def record(self, ids, states, cache, start):
        self.cache = cache
        first = max(0, self.prompt_length - start)
        if first >= ids.shape[1]:
            return
        # A new forward at an earlier position invalidates the old suffix,
        # even if the token ids in that suffix happen to be identical.
        self.rows = {p: row for p, row in self.rows.items() if p < start}
        for i, token in enumerate(ids[0, first:].tolist(), start=first):
            self.rows[start + i] = (int(token), states[0, i])

    def commit(self, tokens):
        self.tokens.extend(int(t) for t in tokens)
        self.flush()

    def flush(self, *, final=False):
        import mlx.core as mx

        ids, rows = [], []
        boundary = self.offset(self.cache)
        while self.emitted < len(self.tokens):
            pos = self.prompt_length + self.emitted
            row = self.rows.get(pos)
            if pos >= boundary or row is None or row[0] != self.tokens[self.emitted]:
                break
            ids.append(row[0])
            rows.append(row[1])
            self.rows.pop(pos)
            self.emitted += 1
        if rows:
            values = mx.contiguous(mx.stack(rows))
            mx.eval(values)
            self.callback(ids, values)
        if final and self.emitted != len(self.tokens):
            raise RuntimeError(
                "generation ended with unexecuted committed feature rows"
            )

    def __enter__(self):
        from mtplx.graphbank import compiled_verify_mode

        if compiled_verify_mode() != "off":
            raise ValueError(
                "committed features currently require eager capture_commit verification"
            )
        if getattr(self.model, "_mtplx_feature_stream", None) is not None:
            raise RuntimeError("a feature stream already owns this runtime")
        if not 0 <= self.layer < len(self.model.model.layers):
            raise ValueError("feature layer is outside this model")
        self.model._mtplx_feature_stream = self
        return self

    def __exit__(self, *exc):
        self.model._mtplx_feature_stream = None
        self.rows.clear()
        self.cache = None
