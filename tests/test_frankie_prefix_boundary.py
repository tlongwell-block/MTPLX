"""A short textual rewrite cannot rewind recurrent state by trimming KV."""
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache
import pytest

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank


class Runtime:
    model_path = Path("models/example")
    mtp_enabled = False

    def make_cache(self):
        return [KVCache(), ArraysCache(2)]


@pytest.mark.parametrize("gap", [1, 2, 8])
@pytest.mark.parametrize("boundary", [False, True])
def test_short_rewrite_requires_correct_recurrent_boundary(gap, boundary, monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "1")
    runtime = Runtime()
    cache = runtime.make_cache()
    end = 16 + gap
    cache[0].update_and_fetch(mx.zeros((1, 1, end, 2)), mx.zeros((1, 1, end, 2)))
    cache[1][0] = mx.array([end], dtype=mx.int32)
    bank = SessionBank(max_entries=4, max_bytes=100_000, per_session_max_bytes=100_000)
    rows = []
    if boundary:
        snapshot = CacheSnapshot(
            states=(None, [mx.array([16], dtype=mx.int32), None]),
            meta_states=(None, None),
        )
        rows = [(16, snapshot, mx.zeros((1, 1, 4)))]
    entry = bank.put(
        runtime=runtime, token_ids=list(range(end)), cache=cache,
        logits=None, hidden=None, gdn_boundaries=rows,
    )
    restored = bank.restore_entry_prefix_cache(runtime, entry, 16, mode="clone")
    if not boundary:
        assert restored is None
    else:
        state, _, _, point, _ = restored
        assert point == 16 and state[0].offset == 16
        assert int(state[1][0].item()) == 16
        assert int(cache[1][0].item()) == end
