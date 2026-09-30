from collections import deque
from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np
import pytest
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen3 import Attention, ModelArgs

from mtplx.frankie.speech_window import SpeechWindowCache, slide


def test_evicted_cache_matches_attention_masked_to_retained_positions():
    attention = Attention(ModelArgs(
        model_type="qwen3", hidden_size=32, num_hidden_layers=1,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, rms_norm_eps=1e-6, vocab_size=32,
        max_position_embeddings=4096, rope_theta=10000, tie_word_embeddings=False,
    ))
    full, window = KVCache(), SpeechWindowCache(KVCache())
    x = mx.random.normal((1, 10, 32))
    a = attention(x, mask="causal", cache=full)
    b = attention(x, mask="causal", cache=window)
    assert mx.max(mx.abs(a - b)).item() < 2e-5
    positions = list(range(10))
    for _ in range(12):
        old = window.offset
        prefix = [np.array(v[..., :2, :]) for v in (window.keys, window.values)]
        window.discard(2, 3)
        positions = positions[:2] + positions[5:]
        assert window.offset == old and window.size() == len(positions)
        for expected, value in zip(prefix, (window.keys, window.values)):
            np.testing.assert_array_equal(value[..., :2, :], expected)
        x = mx.random.normal((1, 3, 32))
        mask = np.zeros((3, old + 3), dtype=bool)
        for i in range(3):
            mask[i, positions + list(range(old, old + i + 1))] = True
        a = attention(x, mask=mx.array(mask), cache=full)
        b = attention(x, mask=window.make_mask(3, return_array=False, window_size=None), cache=window)
        assert mx.max(mx.abs(a - b)).item() < 2e-5
        positions += list(range(old, old + 3))


def test_word_limit_evicts_whole_phrases_and_keeps_reference():
    cache = SpeechWindowCache(KVCache())
    values = mx.arange(10, dtype=mx.float32).reshape(1, 1, 10, 1)
    cache.update_and_fetch(values, values)
    model = NS(
        _speech_cache=[cache], _speech_segments=deque([(4, 4), (4, 4)]),
        _context_words=8, _next_words=5, context_words=10, context_rows=64,
        context_bytes=100_000, _voice_prefix=mx.zeros((2, 1)),
        _window_evictions=0, _window_evicted_rows=0,
    )
    slide(model, 3)
    assert model._context_words == 4 and list(model._speech_segments) == [(4, 4)]
    assert cache.offset == 10 and cache.size() == 6
    np.testing.assert_array_equal(cache.keys.reshape(-1), [0, 1, 6, 7, 8, 9])
    assert model._window_evictions == 1 and model._window_evicted_rows == 4
    with pytest.raises(ValueError, match="eviction"):
        cache.discard(2, 5)
