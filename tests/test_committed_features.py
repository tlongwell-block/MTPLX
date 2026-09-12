from types import SimpleNamespace as NS
import pytest
from mtplx.features import CommittedFeatures

mx = pytest.importorskip("mlx.core")


def runtime():
    return NS(model=NS(model=NS(layers=[None] * 20)))


def test_rejected_suffix_and_pending_correction_are_not_emitted(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    received = []
    cache = [NS(offset=8)]
    with CommittedFeatures(
        runtime(), 4, lambda ids, rows: received.extend(zip(ids, rows[:, 0].tolist()))
    ) as stream:
        stream.record(
            mx.array([[1, 2, 3, 4]]),
            mx.array([[[11.0], [12.0], [13.0], [14.0]]]),
            cache,
            4,
        )
        cache[0].offset = 6
        stream.commit([1, 2, 9])
        assert received == [(1, 11.0), (2, 12.0)]
        cache[0].offset = 9
        stream.record(
            mx.array([[9, 5, 6]]), mx.array([[[19.0], [15.0], [16.0]]]), cache, 6
        )
        cache[0].offset = 8
        stream.commit([5, 8])
        assert received == [(1, 11.0), (2, 12.0), (9, 19.0), (5, 15.0)]
        stream.record(mx.array([[8]]), mx.array([[[18.0]]]), cache, 8)
        cache[0].offset = 9
        stream.flush(final=True)
        assert received[-1] == (8, 18.0)


def test_same_id_above_committed_boundary_remains_pending(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    seen = []
    cache = [NS(offset=1)]
    with CommittedFeatures(runtime(), 1, lambda ids, rows: seen.extend(ids)) as stream:
        stream.record(mx.array([[7]]), mx.ones((1, 1, 2)), cache, 1)
        stream.commit([7])
        assert not seen
        with pytest.raises(RuntimeError, match="unexecuted"):
            stream.flush(final=True)
        cache[0].offset = 2
        stream.flush(final=True)
        assert seen == [7]


def test_owner_cleanup_on_failure(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    rt = runtime()
    with pytest.raises(ValueError):
        with CommittedFeatures(rt, 1, lambda *a: None):
            raise ValueError("cancelled")
    assert rt.model._mtplx_feature_stream is None


def test_compiled_feature_path_fails_explicitly(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "on")
    with pytest.raises(ValueError, match="eager"):
        with CommittedFeatures(runtime(), 1, lambda *a: None):
            pass
