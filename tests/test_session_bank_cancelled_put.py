"""A cancelled snapshot must not evict the prefix needed by resumed input."""
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from mtplx import session_bank as module


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("stage", ["before", "snapshot", "cold_enqueue"])
def test_cancelled_put_keeps_previous_prefix(monkeypatch, lazy, stage):
    monkeypatch.setattr(module, "_lazy_snapshot_enabled", lambda: lazy)
    bank = module.SessionBank(max_entries=1, max_bytes=1024, per_session_max_bytes=1024)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=False)
    common = {"runtime": runtime, "cache": [], "logits": None, "hidden": None,
              "session_id": "conversation", "nbytes_override": 512}
    previous = bank.put(token_ids=[1, 2, 3], **common)
    cancelled = Event()
    if stage == "before":
        cancelled.set()
    else:
        name = ("snapshot_cache_lazy_hybrid" if lazy else "snapshot_cache") if stage == "snapshot" else "_enqueue_cold_entry"
        owner = module if stage == "snapshot" else bank
        original = getattr(owner, name)

        def finish_and_cancel(*args, **kwargs):
            result = original(*args, **kwargs)
            cancelled.set()
            return result

        monkeypatch.setattr(owner, name, finish_and_cancel)
    rejected = bank.put(token_ids=[1, 2, 3, 4], abort_check=cancelled.is_set, **common)
    assert rejected is None
    assert list(bank._entries.values()) == [previous]
    assert bank.longest_prefix([1, 2, 3, 5]) is previous


def test_live_reference_cancel_does_not_supersede_history(monkeypatch):
    bank = module.SessionBank(max_entries=1, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=False)
    common = {"runtime": runtime, "logits": None, "hidden": None, "session_id": "conversation"}
    previous = bank.put(token_ids=[1, 2], cache=[], nbytes_override=128, **common)
    cancelled = Event()
    original = module._empty_cache_snapshot

    def snapshot(cache):
        result = original(cache)
        cancelled.set()
        return result

    monkeypatch.setattr(module, "_empty_cache_snapshot", snapshot)
    rejected = bank.put(token_ids=[1, 2, 3], cache=[SimpleNamespace(state=[])],
                        keep_live_ref=True, nbytes_override=1024,
                        abort_check=cancelled.is_set, **common)
    assert rejected is None
    assert bank.longest_prefix([1, 2, 4]) is previous
