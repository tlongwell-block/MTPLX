"""The floor listener reuses an immutable prefix apart from voice/HTTP state."""

import os
from threading import Event
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache
from test_frankie_http_cache import CachedMTPModel, drain, generate, prepare
from test_generation_sustained import _runtime

from mtplx.frankie.completions import Completions
from mtplx.frankie.floor import FLOOR_POLICY
from mtplx.session_bank import SessionBank


@pytest.fixture(autouse=True)
def cpu_and_eager(monkeypatch):
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_SESSION_MIN_REUSE", "1")
    try:
        yield
    finally:
        mx.set_default_device(device)


def test_listener_warm_uses_private_bounded_bank_and_is_idempotent(monkeypatch):
    service = Completions.__new__(Completions)
    service.listener_bank = None
    service.listener_prefixes = set()
    warmed = []
    expected = [FLOOR_POLICY]

    class FakeBank:
        total_nbytes = 123

        def __init__(self):
            self.entries = set()

        def __len__(self):
            return len(self.entries)

    private = FakeBank()

    def construct(**kwargs):
        assert kwargs == {"max_entries": 1, "max_bytes": 512 * 1024**2,
                          "per_session_max_bytes": 512 * 1024**2, "idle_ttl_s": float("inf")}
        return private

    monkeypatch.setattr("mtplx.session_bank.SessionBank", construct)

    def warm(settings, *, session_id, bank):
        assert settings["instructions"] == expected[len(warmed)]
        assert settings["thinking"] == "off" and settings["tools"] == []
        assert session_id == f"listener-prefix-{len(warmed)}" and bank is private
        bank.entries.add(settings["instructions"])
        warmed.append(settings)

    voice_bank = object()
    service.engine = NS(mtp=2, warm=warm, bank=voice_bank)
    assert service.warm_listener() == {"cache_bytes": 123}
    assert service.warm_listener() == {"cache_bytes": 123}
    assert len(warmed) == 1 and service.listener_prefixes == set(expected)
    assert service.engine.bank is voice_bank and service.listener_bank is private


def test_failed_warm_admission_is_not_reported_as_cached():
    service = Completions.__new__(Completions)
    service.listener_prefixes = set()
    service.listener_bank = SessionBank(max_entries=1, max_bytes=1,
                                       per_session_max_bytes=1)
    service.engine = NS(mtp=2, warm=lambda *args, **kwargs: None)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="memory budget"):
            service.warm_listener()
        assert len(service.listener_bank) == 0
        assert not service.listener_prefixes


def test_failed_admission_can_retry_and_then_cache_only_the_floor_policy():
    service = Completions.__new__(Completions)
    service.listener_prefixes = set()
    warmed = []

    class OneEntryBank:
        total_nbytes = 123

        def __init__(self):
            self.entry = None

        def __len__(self):
            return int(self.entry is not None)

        def clear(self):
            self.entry = None

    service.listener_bank = OneEntryBank()

    def warm(settings, *, session_id, bank):
        assert settings["instructions"] == FLOOR_POLICY
        if warmed:  # Simulate admission becoming available on the retry.
            bank.entry = settings["instructions"]
        warmed.append(session_id)

    service.engine = NS(mtp=2, warm=warm)
    with pytest.raises(RuntimeError, match="memory budget"):
        service.warm_listener()
    assert not service.listener_prefixes and len(service.listener_bank) == 0
    assert service.warm_listener() == {"cache_bytes": 123}
    assert service.warm_listener() == {"cache_bytes": 123}
    assert warmed == ["listener-prefix-0"] * 2
    assert service.listener_prefixes == {FLOOR_POLICY}


@pytest.mark.parametrize("previous", [None, "0", "1", "custom-boundary-mode"])
@pytest.mark.parametrize("outcome", ["success", "raised", "budget"])
def test_boundary_capture_disabled_only_during_warm_and_restored_on_every_exit(monkeypatch, previous, outcome):
    key = "MTPLX_GDN_BOUNDARY_CAPTURE"
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)

    def restored():
        assert os.environ.get(key) == previous
        assert (key in os.environ) == (previous is not None)

    class FakeBank:
        def __init__(self):
            self.filled = False

        def __len__(self):
            restored()  # Admission checking must happen outside suppression.
            return int(self.filled)

        def clear(self):
            restored()
            self.filled = False

        @property
        def total_nbytes(self):
            restored()
            return 123

    private = FakeBank()

    def construct(**kwargs):
        restored()  # Even private-bank construction retains the caller's env.
        return private

    monkeypatch.setattr("mtplx.session_bank.SessionBank", construct)
    service = Completions.__new__(Completions)
    service.listener_bank = None
    service.listener_prefixes = set()
    calls = []
    error = ValueError("Synthetic synchronous warmup failure")

    def warm(settings, *, session_id, bank):
        assert os.environ[key] == "0"
        assert bank is private and settings["instructions"] == FLOOR_POLICY
        calls.append(session_id)
        if outcome == "raised":
            raise error
        bank.filled = outcome == "success"

    service.engine = NS(mtp=2, warm=warm)
    restored()
    if outcome == "raised":
        with pytest.raises(ValueError, match="Synthetic synchronous warmup failure") as caught:
            service.warm_listener()
        assert caught.value is error
    elif outcome == "budget":
        with pytest.raises(RuntimeError, match="memory budget"):
            service.warm_listener()
    else:
        assert service.warm_listener() == {"cache_bytes": 123}
        # Already-warmed calls do not enter the temporary environment change.
        assert service.warm_listener() == {"cache_bytes": 123}
    restored()
    assert calls == ["listener-prefix-0"]
    assert service.listener_prefixes == ({FLOOR_POLICY} if outcome == "success" else set())


def test_divergent_listener_inputs_restore_exact_short_recurrent_prefix(monkeypatch):
    class HybridModel(CachedMTPModel):
        def make_cache(self):
            recurrent = ArraysCache(1)
            recurrent[0] = mx.zeros((1, 1), mx.int32)
            return [KVCache(), recurrent]

        def __call__(self, input_ids, *, cache, **kwargs):
            cache[1][0] = cache[1][0] + mx.sum(input_ids)
            return super().__call__(input_ids, cache=cache, **kwargs)

    runtime = _runtime(HybridModel())
    private = SessionBank(max_entries=1, max_bytes=1 << 24,
                          per_session_max_bytes=1 << 24)
    voice_bank = SessionBank(max_entries=2, max_bytes=1 << 24,
                             per_session_max_bytes=1 << 24)
    prefix = [3, 1, 4] * 30  # Below the normal 512-token near-prefix threshold.
    generate(runtime, prefix, private)
    original_entries = dict(private._entries)

    def freeze(value):
        if isinstance(value, mx.array):
            return str(value.dtype), value.shape, value.tolist()
        if isinstance(value, (list, tuple)):
            return tuple(freeze(part) for part in value)
        if hasattr(value, "states"):
            return freeze(value.states), freeze(value.meta_states)
        return value

    def cache_payloads():
        return {key: freeze((entry.cache_snapshot, entry.mtp_history_snapshot,
                             entry.logits, entry.hidden)) for key, entry in private._entries.items()}

    original_payloads = cache_payloads()
    original_bytes = private.total_nbytes
    service = Completions.__new__(Completions)
    service.engine = NS(runtime=runtime, bank=voice_bank, mtp=2,
                        tokenizer=NS(eos_token_ids=set()))
    service.emit_text = lambda *args: None

    class Detokenizer:
        last_segment = ""

        def add_token(self, token):
            pass

    def forbidden_put(**kwargs):
        pytest.fail("An observation cannot replace the fixed listener prefix.")

    monkeypatch.setattr(private, "put", forbidden_put)
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1")
    for i, suffix in enumerate(([2] * 20, [6] * 25, [7] * 30)):
        ids = prefix + suffix
        job = NS(ids=ids, id=f"listener-{i}", splice=None, bank=private, internal=True,
                 prefill_step_size=32, cancelled=Event(), tokens=[],
                 detokenizer=Detokenizer(), data={"thinking": "off", "max_tokens": 6,
                 "seed": i, "temperature": 0.7, "top_p": 1, "top_k": 0})
        drain(service.mtp_steps(job))
        assert job.cached_tokens == len(prefix)
        expected, _ = generate(runtime, ids, None, seed=i)
        assert job.tokens == expected.tokens
        state, _ = drain(prepare(runtime, prefix, private))
        assert state.cached_tokens == len(prefix)
        assert state.trunk_cache[1][0].item() == sum(prefix)
        assert private._entries == original_entries
        assert cache_payloads() == original_payloads
        assert private.total_nbytes == original_bytes
        assert len(voice_bank) == 0
