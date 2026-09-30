"""Phrase-boundary eviction over the existing MLX KV cache.

RoPE positions continue monotonically while attention uses the retained physical
rows. The immutable voice prefix remains at its original positions. This is
bounded attention history, not an exact recomputation without the evicted past.
"""
import mlx.core as mx


class SpeechWindowCache:
    def __init__(self, cache):
        self.cache = cache
        self.offset = cache.offset

    @property
    def keys(self):
        return self.cache.keys

    @property
    def values(self):
        return self.cache.values

    @property
    def step(self):
        return self.cache.step

    def size(self):
        return self.cache.size()

    def make_mask(self, *args, **kwargs):
        return self.cache.make_mask(*args, **kwargs)

    def update_and_fetch(self, keys, values):
        result = self.cache.update_and_fetch(keys, values)
        self.offset += keys.shape[2]
        return result

    def discard(self, prefix_rows, rows):
        if not 0 <= prefix_rows <= prefix_rows + rows <= self.size():
            raise ValueError('Invalid speech cache eviction')
        if not rows:
            return
        # Materialize compact storage; keeping slices could retain the old buffer.
        state = tuple(mx.contiguous(mx.concatenate(
            [a[..., :prefix_rows, :], a[..., prefix_rows + rows:self.size(), :]],
            axis=2)) for a in (self.keys, self.values))
        self.cache.state = state


def slide(model, required_rows):
    """Evict complete oldest phrases for the next prompt and reserved frames."""
    cache = model._speech_cache
    if not cache:
        return
    physical = cache[0].size()
    bytes_per_row = sum((c.keys.nbytes + c.values.nbytes) // c.keys.shape[2] for c in cache)
    step = cache[0].step
    max_rows = min(model.context_rows, model.context_bytes // (bytes_per_row * step) * step)
    removed = 0
    while model._speech_segments and (
        model._context_words + model._next_words > model.context_words
        or physical - removed + required_rows > max_rows
    ):
        words, rows = model._speech_segments.popleft()
        model._context_words -= words
        removed += rows
    if physical - removed + required_rows > max_rows:
        max_frames = model._max_frames
        model.reset_speech_context()
        model._max_frames = max_frames
        return
    if removed:
        for state in cache:
            state.discard(model._voice_prefix.shape[0], removed)
        mx.eval([(c.keys, c.values) for c in cache])
        model._window_evictions += 1
        model._window_evicted_rows += removed
