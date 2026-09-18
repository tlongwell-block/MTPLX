"""The listener reuses one immutable prefix without polluting voice/HTTP state."""

from threading import Event
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache
from test_frankie_http_cache import CachedMTPModel, drain, generate, prepare
from test_generation_sustained import _runtime

from mtplx.frankie.completions import Completions
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


def test_listener_warm_uses_private_bounded_bank_and_is_idempotent():
    service = Completions.__new__(Completions)
    service.listener_bank = None
    warmed = []

    class FakeBank:
        total_nbytes = 123
        filled = False

        def __len__(self):
            return int(self.filled)

    private = FakeBank()

    def warm(settings, *, session_id, bank):
        assert "exactly one word" in settings["instructions"]
        assert settings["thinking"] == "off" and settings["tools"] == []
        assert session_id == "listener-prefix" and bank is private
        bank.filled = True
        warmed.append(settings)

    service.listener_bank = private
    service.engine = NS(mtp=2, warm=warm, bank=object())
    assert service.warm_listener() == {"cache_bytes": 123}
    assert service.warm_listener() == {"cache_bytes": 123}
    assert len(warmed) == 1


def test_failed_warm_admission_is_not_reported_as_cached():
    service = Completions.__new__(Completions)
    service.listener_bank = SessionBank(max_entries=1, max_bytes=1,
                                       per_session_max_bytes=1)
    service.engine = NS(mtp=2, warm=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="memory budget"):
        service.warm_listener()


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
    original_entries = tuple(private._entries)
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
        assert tuple(private._entries) == original_entries
        assert private.total_nbytes == original_bytes
        assert len(voice_bank) == 0
