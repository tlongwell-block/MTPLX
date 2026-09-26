"""Independent ear-only interruption lifecycle checks, without model inference.

The fake recognizer supplies text, so these qualify ownership and scheduling,
not acoustic accuracy or actual interruption latency.
"""
import asyncio
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_frankie_background_session import call, output, setup, user, wait_for
from test_frankie_prefix_adapter import fixture, snapshot
from test_frankie_regeneration import events, staged_reply
from test_frankie_streaming_session import Listener, clear_events, enable, feed

from mtplx.frankie.backchannel import decide_backchannel
from mtplx.frankie.listening import PrefixEvidence


def forbidden(*args, **kwargs):
    pytest.fail("Ear-only interruption must not invoke the brain or semantic revalidation.")


class FastListener(Listener):
    def __init__(self, text="Please stop.", *, final_text=None, stable=True, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.stable = stable
        self.final_text = text if final_text is None else final_text
        self.service.warm_listener = forbidden

    classify_prefix = recheck_prefix = classify = forbidden

    async def hear_prefix(self, ticket):
        self.snapshots.append(ticket)
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                if not self.ignore_cancel:
                    raise
        if self.error:
            raise self.error
        return PrefixEvidence(ticket, self.text, self.text if self.stable else "",
                              ticket.samples - ticket.sample_rate * 160 // 1000), {"fixture": True}

    async def classify_backchannel(self, item, run, *, fragments=None):
        self.final_items.append(item)
        fragments = fragments or (item,)
        for fragment in fragments:
            for part in fragment["content"]:
                if part["type"] == "input_audio":
                    part["_rows"], part["_transcript"] = object(), self.final_text
        decision = decide_backchannel(
            self.final_text, final=True,
            voiced_ms=sum(i.get("_speech_voiced_ms", 0) for i in fragments),
            elapsed_ms=sum(i.get("_speech_elapsed_ms", 0) for i in fragments))
        action = {"preserve": "continue", "wait": "wait", "yield": "yield"}[decision.action]
        return NS(action=action), NS(user_text=self.final_text), {"fixture": True}


async def fast(s, listener, *, interrupt=True):
    await enable(s, listener, mode="backchannel", interrupt=interrupt)


@pytest.mark.parametrize("text", ["Yeah.", "Mhm.", "Hmm.", "Um, go on."])
def test_brief_nod_preserves_old_audio_through_ear_only_final(text):
    async def check(s, e):
        listener = FastListener(text)
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        assert not old.abort.is_set() and not old.playback_paused and not e.calls
        await feed(s, 0, 10)
        await wait_for(lambda: len(listener.final_items) == 1 and not s.semantic_pending)
        assert not old.abort.is_set() and not e.calls and not s.listening
        assert clear_events(s) == []
        assert not listener.rechecks and not listener.contexts
    asyncio.run(setup(check))


@pytest.mark.parametrize("text", ["Stop.", "Yeah but", "Actually Thursday.",
                                  "Alex, turn left.", "I'm quoting a line."])
def test_first_recognized_non_nod_yields_without_brain_or_recheck(text):
    async def check(s, e):
        listener = FastListener(text)
        await fast(s, listener)
        old = staged_reply(s)
        assert old.done and not old.playback_finished  # Queued PCM is still interruptible.
        await feed(s, 16000, 9)
        assert not listener.snapshots and not old.abort.is_set()
        await feed(s, 16000, 1)
        await wait_for(lambda: old.abort.is_set())
        assert len(listener.snapshots) == 1 and not listener.rechecks
        assert s.listening and s.prefix_yielded and not e.calls and not s.spec
        assert old.text == "The first point."
        assert len(clear_events(s)) == 1
    asyncio.run(setup(check))


@pytest.mark.parametrize("misheard", ["Im.", "I hope"])
def test_unstable_non_nod_waits_then_recognized_nod_preserves_and_stable_interrupt_yields(misheard):
    async def check(s, e):
        listener = FastListener(misheard, stable=False)
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        assert not old.abort.is_set() and s.backchannel_text == ""
        first = events(s)
        observed = next(event for event in first if event["type"] == "frankie.interaction.prefix")
        assert observed["action"] == "wait" and observed["reason"] == "unstable_words"
        assert not observed["apply"]
        assert not any(event["type"] == "frankie.playback.clear" for event in first)
        # An unstable word must not leak into the cheap per-frame deadline gate.
        await feed(s, 16000, 1)
        assert not old.abort.is_set() and clear_events(s) == []
        listener.text = "Mhm, go on."
        await feed(s, 16000, 4)
        await wait_for(lambda: len(listener.snapshots) == 2 and s.prefix_task is None)
        assert not old.abort.is_set() and s.backchannel_text == "Mhm, go on."
        assert clear_events(s) == []  # A recognized nod need not be stable.
        listener.text, listener.stable = "Actually, Thursday.", True
        await feed(s, 16000, 5)
        await wait_for(lambda: old.abort.is_set())
        assert len(listener.snapshots) == 3 and len(clear_events(s)) == 1
        assert s.listening and not e.calls and not listener.rechecks
    asyncio.run(setup(check))


def test_early_yield_prefills_at_normal_pause_but_publishes_only_at_endpoint():
    async def check(s, e):
        listener = FastListener()
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: old.abort.is_set())
        await s.handle({"type": "response.create"})  # Stale harness trigger.
        await feed(s, 17000, 6)
        await feed(s, 0, 2)
        assert s.listening and not e.calls and not s.spec
        await feed(s, 0, 1)
        await wait_for(lambda: len(e.calls) == 1)
        tentative = s.spec
        assert tentative is s.current and not tentative.visible and not tentative.ready.is_set()
        assert s.listening and not any(event["type"].startswith("response.") for event in events(s))
        inputs = [i for i in e.calls[0] if i.get("role") == "user"]
        expected = np.concatenate([np.full(7680, 16000), np.full(4608, 17000), np.zeros(2304)]) / 32768
        np.testing.assert_array_equal(inputs[0]["content"][0]["_pcm"], expected)
        assistant = next(i for i in e.calls[0] if i.get("role") == "assistant")
        assert assistant["content"] == [{"type": "output_audio", "transcript": "The first point."}]
        assert len(inputs) == 1 and not listener.final_items
        await feed(s, 0, 6)
        assert not tentative.visible and s.listening
        assert not any(event["type"].startswith("response.") for event in events(s))
        await feed(s, 0, 1)
        await wait_for(lambda: bool(tentative.text))
        assert s.current is tentative and tentative.visible and not s.spec and not s.listening
        assert not tentative.abort.is_set() and len(e.calls) == 1
        assert any(event["type"] == "response.created" for event in events(s))
    asyncio.run(setup(check))


def test_resumed_speech_cancels_hidden_prefill_and_fresh_prefill_keeps_both_segments():
    async def check(s, e):
        listener = FastListener()
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: old.abort.is_set())
        await feed(s, 0, 3)
        await wait_for(lambda: len(e.calls) == 1)
        abandoned = s.spec
        assert not abandoned.visible and not abandoned.ready.is_set()
        assert not any(event["type"].startswith("response.") for event in events(s))
        await feed(s, 17000, 5)
        await wait_for(lambda: abandoned.done)
        assert abandoned.abort.is_set() and not abandoned.visible and not s.spec
        assert not any(i.get("id") == abandoned.item_id for i in s.items)
        assert s.listening and len(e.calls) == 1
        await feed(s, 0, 3)
        await wait_for(lambda: len(e.calls) == 2)
        replacement = s.spec
        assert replacement is not abandoned and not replacement.visible
        inputs = [i for i in e.calls[1] if i.get("role") == "user"]
        expected = np.concatenate([np.full(7680, 16000), np.zeros(2304),
                                   np.full(3840, 17000), np.zeros(2304)]) / 32768
        assert len(inputs) == 1
        np.testing.assert_array_equal(inputs[0]["content"][0]["_pcm"], expected)
        assert next(i for i in e.calls[1] if i.get("role") == "assistant")["content"] == [
            {"type": "output_audio", "transcript": "The first point."}]
        assert not any(event["type"].startswith("response.") for event in events(s))
        await feed(s, 0, 7)
        await wait_for(lambda: bool(replacement.text))
        assert replacement.visible and not replacement.abort.is_set()
        assert not s.listening and not s.spec and len(e.calls) == 2
    asyncio.run(setup(check))


def test_speculative_text_audio_and_tool_call_stay_private_until_endpoint(monkeypatch):
    monkeypatch.setattr("mtplx.frankie.thinking.public_tool_calls", lambda *a, **kw: [{
        "id": "new_lookup", "function": {"name": "lookup", "arguments": "{}"}}])

    async def check(s, e):
        listener = FastListener()
        await fast(s, listener)
        await s.handle({"type": "session.update", "session": {"output_modalities": ["audio"]}})
        original = e.respond

        def respond(history, settings, emit, abort, **kwargs):
            def with_audio(kind, value):
                emit(kind, value)
                if kind == "text":
                    emit("audio", np.zeros(240, dtype=np.float32))
            return original(history, settings, with_audio, abort, **kwargs)

        e.respond = respond
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: old.abort.is_set())
        await feed(s, 0, 3)
        await wait_for(lambda: len(e.calls) == 1)
        tentative = s.spec
        e.releases[0].set()
        await asyncio.sleep(.02)
        assert not tentative.done and not tentative.visible
        assert not s.task_ledger.tasks
        assert not any(event["type"].startswith("response.") for event in events(s))
        await feed(s, 0, 7)
        await wait_for(lambda: tentative.done)
        published = events(s)
        types = [event["type"] for event in published]
        assert "response.output_audio.delta" in types
        assert "response.output_audio_transcript.delta" in types
        assert types.count("response.function_call_arguments.done") == 1
        assert s.task_ledger.tasks["new_lookup"].status == "running"
    asyncio.run(setup(check))


@pytest.mark.parametrize("text,frames", [("", 50), ("Yeah", 51)])
def test_duration_cap_yields_without_waiting_for_another_ear_result(text, frames):
    async def check(s, e):
        listener = FastListener(text)
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        listener.release.clear()
        await feed(s, 16000, frames - 11)
        assert not old.abort.is_set()
        await feed(s, 16000, 1)
        assert old.abort.is_set() and s.listening and not e.calls
        assert sum(len(p) for p in s.frames) == frames * 768
        assert len(clear_events(s)) == 1
    asyncio.run(setup(check))


@pytest.mark.parametrize("text", ["", "Yeah."])
def test_vap_held_trailing_silence_does_not_age_a_brief_nod_into_interruption(text):
    async def check(s, e):
        listener = FastListener(text)
        await fast(s, listener)
        old = staged_reply(s)
        release = False
        s.turn = NS(append=lambda *args: None, release=lambda *args: release,
                    reset=lambda: None, close=lambda: None)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        await feed(s, 0, 90)
        await wait_for(lambda: s.prefix_task is None)
        assert s.listening and not old.abort.is_set() and not e.calls
        assert s.speech_voiced_ms == 320 and clear_events(s) == []
        release = True
        await feed(s, 0, 1)
        await wait_for(lambda: len(listener.final_items) == 1 and not s.semantic_pending)
        assert not old.abort.is_set() and not e.calls
        assert listener.final_items[0]["_speech_elapsed_ms"] == 320
        assert len(listener.final_items[0]["content"][0]["_pcm"]) == 101 * 768
        assert clear_events(s) == []
    asyncio.run(setup(check))


@pytest.mark.parametrize("text", ["Stop.", ""])
def test_disabled_interrupt_preserves_playback_during_prefix_and_deadline(text):
    async def check(s, e):
        listener = FastListener(text)
        await fast(s, listener, interrupt=False)
        old = staged_reply(s)
        await feed(s, 16000, 24)
        await wait_for(lambda: s.prefix_task is None)
        assert not old.abort.is_set() and not e.calls and clear_events(s) == []
    asyncio.run(setup(check))


@pytest.mark.parametrize("text,should_yield", [("Stop.", True), ("Mhm.", False), ("", False)])
def test_short_completed_input_uses_ear_only_final_path(text, should_yield):
    async def check(s, e):
        listener = FastListener(text, held=True)
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 3)
        await feed(s, 0, 10)
        await wait_for(lambda: len(listener.final_items) == 1 and not s.semantic_pending)
        assert old.abort.is_set() is should_yield
        if should_yield:
            await wait_for(lambda: len(e.calls) == 1)
        assert len(e.calls) == int(should_yield)
        assert listener.final_items[0]["_speech_voiced_ms"] == 96
        assert len(listener.final_items[0]["content"][0]["_pcm"]) == 13 * 768
        assert len(clear_events(s)) == int(should_yield)
    asyncio.run(setup(check))


@pytest.mark.parametrize("error", [False, True])
def test_disabled_interrupt_also_preserves_playback_at_final_endpoint_or_failure(error):
    class FinalListener(FastListener):
        async def classify_backchannel(self, *args, **kwargs):
            if error:
                self.final_items.append(args[0])
                raise RuntimeError("Final fake ear failure")
            return await super().classify_backchannel(*args, **kwargs)

    async def check(s, e):
        listener = FinalListener(held=True)
        await fast(s, listener, interrupt=False)
        old = staged_reply(s)
        await feed(s, 16000, 3)
        await feed(s, 0, 10)
        await wait_for(lambda: len(listener.final_items) == 1 and not s.semantic_pending)
        assert not old.abort.is_set() and not old.playback_paused
        assert s.current is old and not e.calls and clear_events(s) == []
    asyncio.run(setup(check))


def test_late_cancelled_ear_result_cannot_clear_replacement():
    async def check(s, e):
        listener = FastListener(held=True, ignore_cancel=True)
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await listener.entered.wait()
        await s.handle({"type": "input_audio_buffer.clear"})
        await user(s, "A separate question.")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        replacement = s.current
        clear_events(s)
        listener.release.set()
        await wait_for(lambda: s.prefix_task is None)
        assert replacement is not old and not replacement.abort.is_set()
        assert clear_events(s) == [] and s.prefix_listener.inflight is None
    asyncio.run(setup(check))


def test_failed_prefix_leaves_full_input_available_for_final_ear_handling():
    async def check(s, e):
        listener = FastListener(error=RuntimeError("Temporary fake ear failure"))
        await fast(s, listener)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: len(listener.snapshots) == 1 and s.prefix_task is None)
        assert not old.abort.is_set() and clear_events(s) == []
        await feed(s, 0, 10)
        await wait_for(lambda: len(e.calls) == 1)
        assert old.abort.is_set() and len(listener.final_items) == 1
        assert len(listener.final_items[0]["content"][0]["_pcm"]) == 20 * 768
    asyncio.run(setup(check))


def test_queued_tool_result_waits_for_full_user_turn_and_is_spoken_once():
    async def check(s, e):
        listener = FastListener()
        await fast(s, listener)
        await call(s)
        old = staged_reply(s)
        await feed(s, 16000, 10)
        await wait_for(lambda: old.abort.is_set())
        await output(s, value="FAST_REVIEW_LOOKUP_RESULT")
        await s.handle({"type": "response.create"})
        await feed(s, 17000, 4)
        assert s.queued_task_response and not e.calls and s.listening
        await feed(s, 0, 3)
        await wait_for(lambda: len(e.calls) == 1)
        assert "FAST_REVIEW_LOOKUP_RESULT" not in str(e.calls[0])
        assert s.spec is not None and not s.spec.visible and s.queued_task_response
        assert not s.task_ledger.tasks["lookup_1"].consumed
        assert not any(event["type"].startswith("response.") for event in events(s))
        await feed(s, 0, 7)
        assert s.spec is None and s.current.visible
        e.releases[0].set()
        await wait_for(lambda: len(e.calls) == 2)
        assert str(e.calls[1]).count("FAST_REVIEW_LOOKUP_RESULT") == 1
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        assert len(e.calls) == 2
    asyncio.run(setup(check))


@pytest.mark.parametrize("mtp", [0, 2])
def test_real_fast_adapter_compares_two_ear_prefixes_in_one_owner_job_without_brain(mtp):
    async def check(backend, service, session, ear, owner):
        service.prepare = service.warm_listener = service.insert = forbidden
        snap = snapshot()
        evidence, stats = await backend.hear_prefix(snap)
        assert evidence.snapshot is snap and evidence.text == "Stop."
        assert evidence.previous_text == "Stop." and evidence.stable(160)
        assert [len(call[0]) for call in ear.calls] == [240 * 24, 400 * 24]
        assert ear.calls[0][2] == ear.calls[1][2] != threading.get_ident()
        assert stats["listener_owner_slices"] == 1
        assert "completion_tokens" not in stats and service.listener_bank is None
        assert not service.jobs and not service.pending and backend.job is None
    asyncio.run(fixture(check, ["Stop.", "Stop."], mtp=mtp, real_service=True))


@pytest.mark.parametrize("mtp", [0, 2])
def test_real_fast_final_uses_complete_pcm_once_and_reuses_cached_ear_rows(mtp):
    async def check(backend, service, session, ear, owner):
        service.prepare = service.warm_listener = service.insert = forbidden
        pcm = np.linspace(0, .5, 2400, dtype=np.float32)
        part = {"type": "input_audio", "_pcm": pcm, "_rate": 24000}
        item = {"id": "final", "content": [part], "_speech_voiced_ms": 100, "_speech_elapsed_ms": 100}
        for _ in range(2):
            decision, observation, stats = await backend.classify_backchannel(item, NS())
            assert decision.action == "yield" and observation.user_text == "Please stop."
            assert "completion_tokens" not in stats and stats["listener_owner_slices"] == 1
        assert len(ear.calls) == 1 and ear.calls[0][2] != threading.get_ident()
        np.testing.assert_array_equal(ear.calls[0][0], pcm)
        assert "_rows" in part and part["_transcript"] == "Please stop."
        assert not service.jobs and backend.job is None and service.listener_bank is None
    asyncio.run(fixture(check, ["Please stop."], mtp=mtp, real_service=True))


@pytest.mark.parametrize("phase", ["prefix", "final"])
def test_cancellation_inside_real_ear_never_publishes_a_control_result(phase):
    async def check(backend, service, session, ear, owner):
        service.prepare = service.warm_listener = service.insert = forbidden
        ear.hook = backend.cancel
        with pytest.raises(RuntimeError, match="cancelled|consumer"):
            if phase == "prefix":
                await backend.hear_prefix(snapshot())
            else:
                item = {"id": "final", "content": [{"type": "input_audio", "_pcm": np.zeros(2400),
                                                     "_rate": 24000}]}
                await backend.classify_backchannel(item, NS())
        assert len(ear.calls) == 1 and not service.jobs and backend.job is None
        assert service.listener_bank is None
    asyncio.run(fixture(check, ["Stop."], real_service=True))
