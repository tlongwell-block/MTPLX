"""Independent terminal-preparation edge review; no model or tensor operations."""

import asyncio
from types import SimpleNamespace as NS

import pytest
from mtplx.frankie.completions import Completions, Job


def fixture(mtp, callback, *, prepare_only=False, following_http=False):
    marker = object()
    model = NS(_mtplx_feature_stream=marker)
    engine = NS(runtime=NS(model=model), mtp=mtp, background_step=object())
    service = Completions(engine, None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Terminal preparation must not warm or insert the brain.")

    service.warm_listener = service.insert = forbidden
    job = Job({}, True, asyncio.get_running_loop())
    job.internal, job.prepare_only = True, prepare_only
    job.prepare_callback = callback
    service.pending.append(job)
    service.jobs.add(job)
    http = None
    if following_http:
        http = Job({}, True, asyncio.get_running_loop())
        service.pending.append(http)
        service.jobs.add(http)
    return service, job, http, model, marker


def emitted(job):
    result = []
    while not job.events.empty():
        result.append(job.events.get_nowait())
    return result


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("value", [False, 0, "", [], {}])
def test_falsey_internal_terminal_values_retire_and_preserve_pending_http(mtp, value):
    async def check():
        calls = []

        def callback(job):
            assert model._mtplx_feature_stream is None
            calls.append(job)
            return value

        service, job, http, model, marker = fixture(mtp, callback, following_http=True)
        service.step()
        output = emitted(job)
        assert calls == [job] and len(output) == 1
        assert output[0]["prepared"] is value
        assert output[0]["usage"]["listener_owner_slices"] == 1
        assert output[0]["usage"]["listener_owner_work_ms"] >= 0
        assert job.done and job.prepare_callback is None and job.ids is None
        assert service.jobs == {http} and list(service.pending) == [http]
        assert not service.active and service.batch is None
        assert model._mtplx_feature_stream is marker

    asyncio.run(check())


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("outcome", ["cancel_none", "cancel_false", "error"])
def test_callback_cancellation_or_error_never_warms_inserts_or_leaks_features(mtp, outcome):
    async def check():
        def callback(job):
            assert model._mtplx_feature_stream is None
            if outcome == "error":
                raise RuntimeError("Acoustic fixture failure")
            job.cancelled.set()
            return False if outcome == "cancel_false" else None

        service, job, http, model, marker = fixture(mtp, callback, following_http=True)
        service.step()
        output = emitted(job)
        assert output == ([{"error": "Acoustic fixture failure"}] if outcome == "error" else [])
        assert not job.done and job.prepare_callback is None and job.ids is None
        assert job.owner_slices == 1
        assert service.jobs == {http} and list(service.pending) == [http]
        assert not service.active and service.batch is None
        assert model._mtplx_feature_stream is marker

    asyncio.run(check())


@pytest.mark.parametrize("mtp", [0, 2])
def test_explicit_prepare_only_none_result_remains_a_terminal_event(mtp):
    async def check():
        service, job, _, model, marker = fixture(mtp, lambda job: None, prepare_only=True)
        service.step()
        output = emitted(job)
        assert len(output) == 1 and "prepared" in output[0] and output[0]["prepared"] is None
        assert output[0]["usage"]["listener_owner_slices"] == 1
        assert job.done and not service.jobs and not service.pending and not service.active
        assert service.engine.background_step is None
        assert model._mtplx_feature_stream is marker

    asyncio.run(check())
