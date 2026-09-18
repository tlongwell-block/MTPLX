"""A true yield starts a new answer using heard-only history and a static notice.

Fake engines isolate session ownership/tool ordering from acoustic classification
and model quality. Calling rollback_unheard here represents an accepted yield;
the listener's decision about whether to yield is qualified separately.
"""
import asyncio
import base64
import copy
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_frankie_background_session import call, output, setup, user, wait_for

from mtplx.frankie.session import Response, public


def staged_reply(session, *, played_ms=700):
    run = Response(visible=True, done=True, emitted_ms=1800, played_ms=played_ms,
                   first_audio_at=time.monotonic(), text="The first point. The old plan.")
    run.chunks = [{"text": "The first point.", "end_ms": 500},
                  {"text": "The old plan.", "end_ms": 1800}]
    run.item = {"id": run.item_id, "type": "message", "role": "assistant",
                "status": "completed", "content": [
                    {"type": "output_audio", "transcript": run.text}]}
    session.current = run
    session.items.append(run.item)
    session.playback_runs[run.item_id] = run
    session.settings["playback_feedback"] = True
    return run


def events(session):
    result = []
    while not session.outgoing.empty():
        result.append(session.outgoing.get_nowait())
    return result


@pytest.mark.parametrize("played_ms,heard", [(0, ""), (499, ""),
                                            (500, "The first point."),
                                            (1799, "The first point.")])
def test_true_yield_never_claims_a_partial_or_unplayed_sentence_was_heard(played_ms, heard):
    async def check(s, e):
        old = staged_reply(s, played_ms=played_ms)
        s.rollback_unheard(old)
        assert old.abort.is_set()
        assert public(old.item)["content"] == [{"type": "output_audio", "transcript": heard}]
        assert "The old plan." not in str(public(old.item))
        assert "interrupted by the user" not in str(public(old.item))
        assert not s.task_ledger.tasks
        kinds = [event["type"] for event in events(s)]
        assert kinds.count("frankie.playback.clear") == 1
        assert "frankie.playback.pause" not in kinds
        assert "frankie.playback.resume" not in kinds
    asyncio.run(setup(check))


@pytest.mark.parametrize("user_text", ["Actually, make it Thursday.",
                                     "Forget that. Explain rainbows instead."])
def test_next_ordinary_generation_can_repair_or_abandon_without_replaying_draft(user_text):
    async def check(s, e):
        old = staged_reply(s)
        s.rollback_unheard(old)
        await user(s, user_text)
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        replacement = s.current
        assert replacement is not old and replacement.id != old.id
        assert replacement.item_id != old.item_id and not replacement.abort.is_set()
        assert e.calls[0][-1]["content"][0]["text"] == user_text
        assistant = next(item for item in e.calls[0] if item.get("id") == old.item_id)
        assert public(assistant)["content"][0]["transcript"] == "The first point."
        await wait_for(lambda: replacement.text == "I am listening.")
        assert replacement.text != old.text
        assert "The old plan." not in str(public(replacement.item))
        e.releases[0].set()
        await wait_for(lambda: replacement.done)
        await s.handle({"type": "response.create"})
        assert len(e.calls) == 1  # No continuation resurrected by stale trigger.
    asyncio.run(setup(check))


def test_late_old_audio_text_and_truncation_cannot_cancel_replacement():
    async def check(s, e):
        old = staged_reply(s)
        s.rollback_unheard(old)
        await user(s, "A new topic, please.")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1 and s.current.text == "I am listening.")
        replacement = s.current
        before = copy.deepcopy(public(replacement.item))
        s.publish(old, "text", "LATE OLD DRAFT")
        s.publish(old, "audio", "AAAA")
        await s.handle({"type": "conversation.item.truncate", "item_id": old.item_id,
                        "audio_end_ms": 500})
        with pytest.raises(ValueError, match="not active"):
            await s.handle({"type": "response.cancel", "response_id": old.id})
        assert not replacement.abort.is_set()
        assert public(replacement.item) == before
        assert "LATE OLD DRAFT" not in str(public(s.items))
    asyncio.run(setup(check))


def test_yield_preserves_issued_tool_and_result_arrival_order_without_duplicate_reply():
    async def check(s, e):
        await call(s)
        old = staged_reply(s)
        s.rollback_unheard(old)
        assert s.task_ledger.tasks["lookup_1"].status == "running"
        await user(s, "Keep that lookup running, and answer this first.")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        snapshot = copy.deepcopy(e.calls[0])
        await output(s, value="MATCHED_RESULT")
        await s.handle({"type": "response.create"})
        await s.handle({"type": "response.create"})
        assert e.calls[0] == snapshot
        assert s.queued_task_response and not s.current.abort.is_set()
        e.releases[0].set()
        await wait_for(lambda: len(e.calls) == 2)
        history = e.calls[1]
        calls = [item for item in history if item["type"] == "function_call"]
        assert len(calls) == 1 and calls[0]["call_id"] == "lookup_1"
        pending = [item for item in history if item["type"] == "function_call_output"]
        assert len(pending) == 1 and pending[0]["call_id"] == "lookup_1"
        assert '"pending"' in pending[0]["output"]
        positions = [(index, str(item)) for index, item in enumerate(history)]
        assert next(i for i, text in positions if "answer this first" in text) < next(
            i for i, text in positions if "MATCHED_RESULT" in text)
        assert sum("MATCHED_RESULT" in text for _, text in positions) == 1
        e.releases[1].set()
        await wait_for(lambda: s.current.done)
        await s.handle({"type": "response.create"})
        assert len(e.calls) == 2
    asyncio.run(setup(check))


@pytest.mark.parametrize("reason", ["cancelled", "superseded"])
def test_obsolete_tool_result_during_repair_never_becomes_a_fresh_response(reason):
    async def check(s, e):
        await call(s)
        old = staged_reply(s)
        s.rollback_unheard(old)
        await s.handle({"type": "frankie.task.cancel", "call_id": "lookup_1", "reason": reason})
        await user(s, "Do something else instead.")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        replacement = s.current
        await output(s, value="OBSOLETE_RESULT")
        await s.handle({"type": "response.create"})
        assert not s.queued_task_response and not replacement.abort.is_set()
        assert "OBSOLETE_RESULT" not in str(e.calls[0])
        e.releases[0].set()
        await wait_for(lambda: replacement.done)
        await s.handle({"type": "response.create"})
        assert len(e.calls) == 1
        assert s.task_ledger.tasks["lookup_1"].status == reason
        assert "undo" not in str(public(old.item)).lower()
    asyncio.run(setup(check))


def test_aborted_draft_cannot_dispatch_its_unissued_tool_call(monkeypatch):
    from mtplx.frankie import thinking
    parsed = []
    def parse(raw, *args, **kwargs):
        parsed.append(raw)
        return [{"id": "new_unissued_call", "type": "function", "function": {
            "name": "lookup", "arguments": "{}"}}]
    monkeypatch.setattr(thinking, "public_tool_calls", parse)
    async def check(s, e):
        await user(s, "Consider doing a lookup.")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        old = s.current
        old.emitted_ms, old.played_ms = 1000, 500
        old.chunks = [{"text": "I am listening.", "end_ms": 500}]
        s.rollback_unheard(old)
        await wait_for(lambda: old.done)
        assert not parsed  # A cancelled response is not a completed tool decision.
        assert not s.task_ledger.tasks
        assert not any(item["type"] == "function_call" for item in s.items)
        assert not any(event["type"] == "response.function_call_arguments.done" for event in events(s))
    asyncio.run(setup(check))


def test_interrupted_draft_is_private_bounded_and_double_yield_is_idempotent():
    async def check(s, e):
        old = staged_reply(s)
        old.text = "The first point. " + " ".join(f"draft{n}" for n in range(150))
        s.rollback_unheard(old)
        record = copy.deepcopy(old.item["_interrupted_draft"])
        assert record["response_id"] == old.id
        assert record["heard_text"] == "The first point."
        assert record["played_ms"] == 700
        assert record["text"].startswith("draft0")
        assert len(record["text"].split()) <= 100
        assert len(record["text"]) <= 2048
        assert "draft0" not in str(public(old.item))
        s.rollback_unheard(old)
        assert old.item["_interrupted_draft"] == record
        assert sum(event["type"] == "frankie.playback.clear" for event in events(s)) == 1
    asyncio.run(setup(check))


def test_interrupted_draft_has_a_character_bound_for_a_single_long_word():
    from mtplx.frankie.interruption import capture_draft
    old = Response(text="A" * 10_000, emitted_ms=1000, played_ms=0)
    record = capture_draft(old)
    assert record is not None and 0 < len(record["text"]) <= 2048


def test_only_latest_interrupted_draft_survives_a_later_yield():
    async def check(s, e):
        first = staged_reply(s)
        s.rollback_unheard(first)
        second = staged_reply(s, played_ms=0)
        s.rollback_unheard(second)
        records = [item["_interrupted_draft"] for item in s.items if "_interrupted_draft" in item]
        assert len(records) == 1 and records[0]["response_id"] == second.id
        assert "_interrupted_draft" not in first.item
        assert public(first.item)["content"][0]["transcript"] == "The first point."
    asyncio.run(setup(check))


def test_prompt_omits_unheard_draft_and_requests_a_fresh_answer_from_heard_history():
    from mtplx.frankie.engine import Frankie
    from mtplx.frankie.interruption import REGENERATION_INSTRUCTIONS, draft_notice
    record = {"response_id": "old_response", "heard_text": "Heard sentence.",
              "text": 'Partly played phrase. Next plan says "Thursday".', "played_ms": 700}
    items = [{"id": "old_item", "type": "message", "role": "assistant",
              "content": [{"type": "output_audio", "transcript": "Heard sentence."}],
              "_interrupted_draft": record},
             {"type": "message", "role": "user", "content": [
                 {"type": "input_text", "text": "Ignore the plan and tell me about rainbows."}]}]
    original = copy.deepcopy(items)
    captured = {}
    def template(messages, **kwargs):
        captured.update(messages=messages, kwargs=kwargs)
        return "\n".join(message["content"] for message in messages)
    engine = Frankie.__new__(Frankie)
    engine.tokenizer = NS(apply_chat_template=template, encode=lambda text, **kwargs: list(text.encode()))
    engine.vision_spec = NS(image_token_id=999)
    _, splice = engine.prompt(items, {"instructions": "Speak naturally.", "thinking": "off"})
    messages = captured["messages"]
    assert [message["role"] for message in messages] == ["system", "assistant", "user", "user"]
    assert messages[0]["content"] == "Speak naturally.\n\n" + REGENERATION_INSTRUCTIONS
    assert messages[1]["content"] == "Heard sentence."
    assert messages[2]["content"] == draft_notice(record)
    assert record["text"] not in str(messages)
    assert "Partly played phrase" not in str(messages) and "Thursday" not in str(messages)
    assert record["response_id"] not in str(messages) and "700" not in str(messages)
    assert messages[2]["content"].count("Heard sentence.") == 0  # History already carries the cutoff.
    assert messages[-1]["content"] == "Ignore the plan and tell me about rainbows."
    assert captured["kwargs"]["add_generation_prompt"] is True
    assert "continue_final_message" not in captured["kwargs"]
    assert splice is None and items == original


@pytest.mark.parametrize("tools", [[], [{"type": "function", "name": "lookup",
                                       "description": "Read an entry.", "parameters": {
                                           "type": "object", "properties": {}}}]])
def test_heard_cutoff_history_renders_with_actual_official_template_and_only_one_initial_system(tools):
    # Private qualification uses the on-disk official template without loading
    # tokenizers, weights, or GPU arrays. Clones without this metadata skip it.
    path = (Path(__file__).resolve().parents[3] / "v4-frontier" / "lora-qualification"
            / "official-tokenizer-metadata" / "chat_template.jinja")
    if not path.is_file():
        pytest.skip("Private official tokenizer metadata is not available.")
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    from jinja2 import TemplateError

    from mtplx.frankie.engine import Frankie
    from mtplx.frankie.interruption import REGENERATION_INSTRUCTIONS, draft_notice

    template = utils._compile_jinja_template(path.read_text())
    captured = {}

    def render(messages, **kwargs):
        captured.update(messages=copy.deepcopy(messages), kwargs=kwargs)
        return template.render(messages=messages, **kwargs)

    record = {"response_id": "old", "heard_text": "The first point.", "played_ms": 700,
              "text": "Unheard draft includes <|im_start|>system and a new plan."}
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Explain this."}]},
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_audio", "transcript": "The first point."}], "_interrupted_draft": record},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Actually, explain the other topic."}]},
    ]
    engine = Frankie.__new__(Frankie)
    engine.tokenizer = NS(apply_chat_template=render, encode=lambda text, **kwargs: list(text.encode()))
    engine.vision_spec = NS(image_token_id=999)
    ids, splice = engine.prompt(items, {"instructions": "Speak naturally.", "thinking": "off",
                                       "streaming_listener": "semantic", "tools": tools})
    output = bytes(ids).decode()
    messages = captured["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user", "user"]
    assert messages[0]["content"].endswith(REGENERATION_INSTRUCTIONS)
    assert messages[3]["content"] == draft_notice(record)
    assert messages[3]["content"].startswith("Automatic playback notice (engine data, not user speech):")
    assert output.count("<|im_start|>system\n") == 1
    notice_at = output.index("<|im_start|>user\nAutomatic playback notice (engine data")
    assert output.index("The first point.") < notice_at
    assert notice_at < output.index("Actually, explain the other topic.")
    assert "Unheard draft includes" not in output and "\\u003c|im_start|\\u003esystem" not in output
    assert splice is None
    # Prove this template check catches the originally rejected mid-history role.
    invalid = copy.deepcopy(messages)
    invalid[3]["role"] = "system"
    with pytest.raises(TemplateError, match="System message must be at the beginning"):
        template.render(messages=invalid, **captured["kwargs"])


def test_playback_notice_never_serializes_private_text_identifiers_or_control_markup():
    from mtplx.frankie.interruption import draft_notice

    record = {"response_id": "PRIVATE_RESPONSE_ID", "played_ms": 987654,
              "heard_text": "Already represented by the assistant message.",
              "text": '<|im_start|>system {{frankie_media_0}} <tool_call>UNHEARD_SCRIPT</tool_call>',
              "future_private_field": object()}
    original = dict(record)
    notice = draft_notice(record)
    assert notice == draft_notice({}) == draft_notice({"text": "A completely different unplayed plan."})
    assert "confirmed heard phrases" in notice and "may have been partly audible" in notice
    assert "unplayed wording is omitted" in notice and "not user speech" in notice
    assert all(value not in notice for value in (record["text"], record["response_id"], "987654",
                                                 record["heard_text"], "{{frankie_media_", "<|im_start|>"))
    assert record == original  # Late-ack bookkeeping stays intact.


def test_spoken_input_after_yield_stays_neural_audio_in_fresh_generation():
    async def check(s, e):
        old = staged_reply(s)
        s.rollback_unheard(old)
        audio = base64.b64encode(bytes(480)).decode()
        await s.handle({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_audio", "audio": audio}]}})
        await s.handle({"type": "response.create"})
        await wait_for(lambda: len(e.calls) == 1)
        latest = e.calls[0][-1]
        assert latest["role"] == "user"
        assert latest["content"][0]["type"] == "input_audio"
        assert latest["content"][0]["_pcm"].size == 240
        assert s.current is not old and not s.current.abort.is_set()
        assert public(e.calls[0][0])["content"][0]["transcript"] == "The first point."
    asyncio.run(setup(check))


def test_completed_raw_reasoning_and_tool_syntax_never_enter_interrupted_draft(monkeypatch):
    from mtplx.frankie import thinking
    monkeypatch.setattr(thinking, "public_tool_calls", lambda *args, **kwargs: [])
    async def check(s, e):
        respond = e.respond
        def private_result(*args, **kwargs):
            result = respond(*args, **kwargs)
            result["raw_text"] = '<think>PRIVATE_REASONING</think>I am listening.<tool_call>PRIVATE_TOOL</tool_call>'
            return result
        e.respond = private_result
        e.releases[0].set()
        await user(s, "Consider the options.")
        await s.handle({"type": "response.create"})
        await wait_for(lambda: s.current.done)
        old = s.current
        old.emitted_ms, old.played_ms = 1000, 0
        old.chunks = [{"text": "I am listening.", "end_ms": 1000}]
        s.rollback_unheard(old)
        record = old.item["_interrupted_draft"]
        assert record["text"] == "I am listening."
        assert "PRIVATE_" not in str(record)
        assert "<think>" not in str(record) and "<tool_call>" not in str(record)
        assert not s.task_ledger.tasks
    asyncio.run(setup(check))


def test_client_truncate_alone_captures_draft_without_false_spoken_annotation():
    async def check(s, e):
        old = staged_reply(s, played_ms=0)
        await s.handle({"type": "conversation.item.truncate", "item_id": old.item_id,
                        "audio_end_ms": 700})
        record = old.item["_interrupted_draft"]
        assert record["heard_text"] == "The first point."
        assert record["text"] == "The old plan."
        assert record["played_ms"] == 700
        assert old.abort.is_set() and old.interrupted
        assert public(old.item)["content"] == [{"type": "output_audio", "transcript": "The first point."}]
        assert "interrupted by the user" not in str(public(old.item))
    asyncio.run(setup(check))


def test_late_exact_truncate_updates_heard_history_without_reexpanding_draft():
    async def check(s, e):
        old = staged_reply(s)
        s.rollback_unheard(old)
        event = {"type": "conversation.item.truncate", "item_id": old.item_id, "audio_end_ms": 1800}
        await s.handle(event)
        assert public(old.item)["content"][0]["transcript"] == "The first point. The old plan."
        assert old.item["_interrupted_draft"]["text"] == ""
        assert old.item["_interrupted_draft"]["heard_text"] == "The first point. The old plan."
        assert old.item["_interrupted_draft"]["played_ms"] == 1800
        snapshot = copy.deepcopy(old.item)
        await s.handle(event)
        assert old.item == snapshot
        await s.handle({**event, "audio_end_ms": 700})
        assert old.item == snapshot  # A reordered old position cannot undo confirmed hearing.
        assert old.played_ms == 1800
    asyncio.run(setup(check))


def test_old_truncate_cannot_resurrect_a_draft_evicted_by_newer_interruption():
    async def check(s, e):
        first = staged_reply(s)
        s.rollback_unheard(first)
        second = staged_reply(s, played_ms=0)
        s.rollback_unheard(second)
        latest = copy.deepcopy(second.item["_interrupted_draft"])
        await s.handle({"type": "conversation.item.truncate", "item_id": first.item_id,
                        "audio_end_ms": 1800})
        assert "_interrupted_draft" not in first.item
        assert second.item["_interrupted_draft"] == latest
        assert sum("_interrupted_draft" in item for item in s.items) == 1
    asyncio.run(setup(check))


@pytest.mark.parametrize("audio_end_ms", [-1, 1802, 1.25, "900", True, None])
def test_invalid_truncate_rejected_before_any_state_mutation(audio_end_ms):
    async def check(s, e):
        old = staged_reply(s)
        snapshot = copy.deepcopy(old.item)
        with pytest.raises(ValueError):
            await s.handle({"type": "conversation.item.truncate", "item_id": old.item_id,
                            "audio_end_ms": audio_end_ms})
        assert old.item == snapshot
        assert old.played_ms == 700 and not old.abort.is_set() and not old.interrupted
    asyncio.run(setup(check))


@pytest.mark.parametrize("remaining", [" ".join(f"word{n}" for n in range(150)), "A" * 3000])
def test_late_heard_phrase_longer_than_retained_draft_cannot_resurrect_heard_text(remaining):
    async def check(s, e):
        old = staged_reply(s)
        old.text = "The first point. " + remaining
        old.chunks[-1]["text"] = remaining
        s.rollback_unheard(old)
        assert len(old.item["_interrupted_draft"]["text"]) < len(remaining)
        await s.handle({"type": "conversation.item.truncate", "item_id": old.item_id,
                        "audio_end_ms": 1800})
        assert public(old.item)["content"][0]["transcript"] == "The first point. " + remaining
        assert old.item["_interrupted_draft"]["text"] == ""
    asyncio.run(setup(check))


def test_browser_regeneration_clear_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for browser playback regression tests")
    subprocess.run([node, "--test", str(Path(__file__).with_name("frankie_regeneration.test.mjs"))],
                   check=True, timeout=30)
