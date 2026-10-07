from collections import deque
from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np
import pytest
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen3 import Attention, ModelArgs

from mtplx.frankie.speech_window import SpeechWindowCache, hold, slide


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


def test_hold_keeps_newest_whole_phrases_within_the_word_limit():
    cache = SpeechWindowCache(KVCache())
    values = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
    cache.update_and_fetch(values, values)
    model = NS(_speech_cache=[cache], _speech_segments=deque([(30, 3), (20, 3), (15, 4)]),
               _context_words=65, _voice_prefix=mx.zeros((2, 1)))
    hold(model, 40)
    assert model._context_words == 35 and list(model._speech_segments) == [(20, 3), (15, 4)]
    np.testing.assert_array_equal(cache.keys.reshape(-1), [0, 1, 5, 6, 7, 8, 9, 10, 11])
    hold(model, 40)
    assert cache.size() == 9  # already within the limit
    model._speech_cache = None
    hold(model, 0)  # nothing spoken yet


def test_hold_speech_is_a_reset_when_off_or_without_a_window():
    from mtplx.frankie.breeze import BreezeModel

    for words, mode in ((0, "sliding"), (40, "reset")):
        resets = []
        model = NS(hold_words=words, context_mode=mode, reset_speech_context=lambda: resets.append(True))
        BreezeModel.hold_speech(model)
        assert resets == [True]


def test_guidance_holds_the_same_phrases_in_both_lanes():
    from mtplx.frankie.breeze import BreezeModel
    from mtplx.frankie.demo.guidance import PairedGuidance

    def lane():
        cache = SpeechWindowCache(KVCache())
        values = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        cache.update_and_fetch(values, values)
        return dict(_speech_cache=[cache], _context_words=50, _continuing=True,
                    _speech_segments=deque([(30, 3), (20, 3)]), _chunk_start_rows=5,
                    _window_evictions=0, _window_evicted_rows=0)

    class Model:
        hold_speech = BreezeModel.hold_speech
        hold_words, context_mode, _voice_prefix = 40, "sliding", mx.zeros((2, 1))
        generate = _prompt_embeddings = _depth_tokens = reset_speech_context = staticmethod(lambda *a, **k: None)
        backbone_model = NS(make_cache=lambda: None)

    model = Model()
    model.__dict__.update(lane())
    g = PairedGuidance(model, None, None).install()
    g.shadow = lane()
    model.hold_speech()
    assert g.in_step() and model._context_words == g.shadow["_context_words"] == 20
    assert model._speech_cache[0].size() == g.shadow["_speech_cache"][0].size() == 5
