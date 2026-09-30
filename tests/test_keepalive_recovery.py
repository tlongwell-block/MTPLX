from threading import Event, get_ident

from mtplx.model_scheduler import ModelWorkScheduler


def failing_beat():
    raise RuntimeError("Test keepalive failed")


def test_recovery_runs_once_on_owner_and_does_not_strand_requests():
    scheduler = ModelWorkScheduler(name="test-recovery")
    calls = []
    try:
        scheduler.arm_idle_keepalive(
            failing_beat, interval_s=100, attentive_s=100,
            on_failure=lambda: calls.append(get_ident()),
        )
        for _ in range(2):
            scheduler.submit(scheduler._run_keepalive).result(2)
        assert not calls
        for _ in range(2):
            scheduler.submit(scheduler._run_keepalive).result(2)
        assert calls == [scheduler.owner_thread_id]
        assert not scheduler.keepalive_state()["armed"]
        assert scheduler.submit(lambda: 42).result(2) == 42
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_inflight_old_failure_does_not_disable_rearmed_keepalive():
    scheduler = ModelWorkScheduler(name="test-rearm")
    entered, release = Event(), Event()
    calls = []

    def beat():
        entered.set()
        assert release.wait(2)
        failing_beat()

    try:
        scheduler.arm_idle_keepalive(beat, interval_s=100, attentive_s=100,
                                     on_failure=lambda: calls.append("old"))
        future = scheduler.submit(scheduler._run_keepalive)
        assert entered.wait(1)
        scheduler.arm_idle_keepalive(beat, interval_s=100, attentive_s=100,
                                     on_failure=lambda: calls.append("new"))
        release.set()
        future.result(2)
        for _ in range(2):
            scheduler.submit(scheduler._run_keepalive).result(2)
        assert not calls and scheduler.keepalive_state()["armed"]
        scheduler.submit(scheduler._run_keepalive).result(2)
        assert calls == ["new"]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)
