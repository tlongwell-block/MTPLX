"""Idle work must preserve history and leave foreground decisions authoritative."""
import asyncio
import base64
from threading import Event
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_frankie_session import setup
from mtplx.frankie.session import Response


def ready(s):
    s.history_prefill_enabled = True
    run = Response(visible=True, done=True, playback_finished=True)
    run.item = {"id": run.item_id, "type": "message", "role": "assistant",
                "status": "completed", "content": [
                    {"type": "output_audio", "transcript": "a sentence with several words " * 8}]}
    s.current = run
    s.items.append(run.item)
    s.response_revision = s.user_revision
    return run


@pytest.mark.parametrize("condition", ["disabled", "short", "playing", "pending_input",
                                      "http", "tasks", "failed", "tool"])
def test_ineligible_history_never_submits_owner_work(condition):
    async def check(s):
        run = ready(s)
        if condition == "disabled": s.history_prefill_enabled = False
        if condition == "short": run.item["content"][0]["transcript"] = "Yes."
        if condition == "playing": run.playback_finished = False
        if condition == "pending_input": s.user_revision += 1
        if condition == "http": s.listener = NS(service=NS(jobs={object()}), cancel=lambda: None)
        if condition == "tasks": s.settings["background_tasks"] = True
        if condition == "failed": run.item["status"] = "incomplete"
        if condition == "tool": run.tool_items["call"] = {}
        calls = []
        s.engine.warm = lambda *a, **kw: calls.append(kw)
        s.prepare_idle_history()
        await asyncio.sleep(0)
        assert not calls and s.history_prefill_run is None
    asyncio.run(setup(check))


def test_history_snapshot_is_frozen_and_repeated_feedback_does_not_repeat_work():
    async def check(s):
        run = ready(s)
        user = {"role": "user", "content": [{"type": "input_audio", "_rows": object()}]}
        s.items.insert(0, user)
        calls = []
        s.engine.warm = lambda settings, **kw: calls.append((settings, kw))
        original = run.item["content"][0]["transcript"]
        s.prepare_idle_history()
        run.item["content"][0]["transcript"] = "Later playback edit."
        s.prepare_idle_history()
        await asyncio.gather(*s.tasks)
        assert len(calls) == 1
        settings, kw = calls[0]
        assert settings is not s.settings
        assert kw["history"][0] is user
        assert kw["history"][1]["content"][0]["transcript"] == original
        assert kw["prefill_step_size"]() == 32
    asyncio.run(setup(check))


@pytest.mark.parametrize("action", ["voice", "new_request", "http", "settings", "close"])
def test_foreground_arrival_aborts_unfinished_warm(action):
    async def check(s):
        ready(s)
        entered, aborted = Event(), Event()
        def warm(settings, **kw):
            check_abort = kw.get("abort_check")
            if check_abort is None: return  # Ordinary session-prefix warmup.
            entered.set()
            while not check_abort():
                if aborted.wait(.001): return
            aborted.set()
        s.engine.warm = warm
        s.prepare_idle_history()
        assert await asyncio.to_thread(entered.wait, 2)
        if action == "voice":
            pcm = np.full(768, 16000, dtype="<i2")
            await s.receive_audio(base64.b64encode(pcm).decode())
        if action == "new_request": s.current = Response()
        if action == "http": s.listener = NS(service=NS(jobs={object()}), cancel=lambda: None)
        if action == "settings":
            await s.handle({"type": "session.update", "session": {"instructions": "Updated."}})
        if action == "close": s.closed = True
        assert await asyncio.to_thread(aborted.wait, 2)
        await asyncio.gather(*s.tasks)
    asyncio.run(setup(check))
