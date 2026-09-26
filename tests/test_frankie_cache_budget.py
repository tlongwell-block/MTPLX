"""Realtime cache settings preserve defaults and honor bounded operator overrides."""

from types import SimpleNamespace

import pytest

from mtplx.frankie import engine


@pytest.fixture
def make_bank(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    monkeypatch.setattr(engine, "load", lambda *a, **k: SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(engine, "AudioModels", lambda *a: None)
    monkeypatch.setattr(engine, "load_vision_tower", lambda *a: None)
    monkeypatch.setattr(engine, "vision_spec_for_model_dir", lambda *a: None)
    (tmp_path / "preprocessor_config.json").write_text("{}")
    return lambda: engine.Frankie(tmp_path, tmp_path).bank


def test_defaults_preserve_eight_gib(make_bank):
    bank = make_bank()
    assert bank.max_bytes == bank.per_session_max_bytes == 8 * 1024**3
    assert bank.max_entries == 6


def test_standard_byte_overrides_and_total_cap(monkeypatch, make_bank):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "12GiB")
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "16G")
    bank = make_bank()
    assert bank.max_bytes == bank.per_session_max_bytes == 12 * 1024**3


def test_independent_per_session_cap(monkeypatch, make_bank):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "16G")
    bank = make_bank()
    assert bank.max_bytes == 16 * 1024**3
    assert bank.per_session_max_bytes == 8 * 1024**3


def test_invalid_override_warns_and_keeps_default(monkeypatch, make_bank, caplog):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "invalid")
    bank = make_bank()
    assert bank.max_bytes == 8 * 1024**3
    assert "MTPLX_SESSION_BANK_MAX_BYTES" in caplog.text
