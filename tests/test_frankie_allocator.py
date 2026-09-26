"""The normal server's allocator policy also runs before Frankie model loading."""

import argparse
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mtplx.frankie import server


@pytest.fixture(autouse=True)
def isolate_startup_environment():
    # The real sustained profile configures process-wide environment variables.
    with patch.dict(os.environ):
        yield


@pytest.mark.parametrize("environment,cli,expected", [
    (None, [], 8),
    ("4G", [], 4),
    ("4G", ["--mlx-cache-limit", "2G"], 2),
    ("4G", ["--mlx-cache-limit", "off"], None),
])
def test_allocator_policy_precedes_model_load(monkeypatch, environment, cli, expected):
    import mlx.core as mx
    import uvicorn

    from mtplx.frankie import completions, engine
    from mtplx.server import openai

    calls = []
    monkeypatch.setenv("MTPLX_FRANKIE_TOKEN", "test-only")
    monkeypatch.delenv("MTPLX_MEMORY_BUDGET", raising=False)
    if environment is None:
        monkeypatch.delenv("MTPLX_MLX_CACHE_LIMIT", raising=False)
    else:
        monkeypatch.setenv("MTPLX_MLX_CACHE_LIMIT", environment)
    monkeypatch.setattr(openai, "_total_ram_bytes", lambda: 256 * 1024**3)
    monkeypatch.setattr(mx, "set_default_device", lambda _: None)
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: calls.append(n) or 192 * 1024**3)

    def load(*args, **kwargs):
        assert calls == ([] if expected is None else [expected * 1024**3])
        calls.append("loaded")
        return SimpleNamespace(respond=lambda *a, **k: None,
                               bank=SimpleNamespace(clear=lambda: None))

    async def startup(app):
        async with app.router.lifespan_context(app):
            pass

    monkeypatch.setattr(engine, "Frankie", load)
    monkeypatch.setattr(completions, "attach_routes", lambda *a, **k: None)
    monkeypatch.setattr(uvicorn, "run", lambda app, **k: asyncio.run(startup(app)))
    args = server.add_arguments(argparse.ArgumentParser()).parse_args(
        ["--brain", "model", "--audio", "audio", *cli])
    assert server.serve(args) == 0
    assert calls[-1] == "loaded"
