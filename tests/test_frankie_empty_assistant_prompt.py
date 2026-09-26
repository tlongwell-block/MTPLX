"""Empty heard history is omitted from model input, not rewritten or retried."""

import asyncio
import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("mlx.core")

from mtplx.frankie.engine import Frankie
from mtplx.frankie.interruption import draft_notice


def message(role, text, **extra):
    return {"type": "message", "role": role, "content": [
        {"type": "output_audio" if role == "assistant" else "input_text",
         "transcript" if role == "assistant" else "text": text}], **extra}


def render(items, *, actual_template=False):
    captured = {}
    template = None
    if actual_template:
        path = Path(os.environ.get("FRANKIE_TEST_CHAT_TEMPLATE", ""))
        if not path.is_file():
            pytest.skip("Set FRANKIE_TEST_CHAT_TEMPLATE to the brain tokenizer chat_template.jinja.")
        utils = pytest.importorskip("transformers.utils.chat_template_utils")
        template = utils._compile_jinja_template(path.read_text())

    def apply(messages, **kwargs):
        captured["messages"] = copy.deepcopy(messages)
        assert kwargs["preserve_thinking"] is False
        if template is not None:
            return template.render(messages=messages, **kwargs)
        return "\n".join(str(item) for item in messages)

    engine = Frankie.__new__(Frankie)
    engine.tokenizer = NS(apply_chat_template=apply,
                          encode=lambda text, **kwargs: list(text.encode()))
    engine.vision_spec = NS(image_token_id=999)
    original = copy.deepcopy(items)
    ids, splice = engine.prompt(items, {"instructions": "Speak naturally.", "thinking": "off"}, public_history=True)
    assert items == original and splice is None
    return captured["messages"], bytes(ids).decode()


@pytest.mark.parametrize("text", ["", " \n\t"])
@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_empty_assistant_is_omitted_only_from_model_view(text, status):
    items = [message("user", "First question."), message("assistant", text, status=status),
             message("user", "Next question.")]
    messages, _ = render(items)
    assert [item["role"] for item in messages] == ["system", "user", "user"]
    assert len(items) == 3 and items[1]["status"] == status


def test_tool_call_with_empty_content_stays_substantive_and_correlated():
    messages, _ = render([
        message("user", "Look it up."), message("assistant", ""),
        {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call-1", "output": "42"}])
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["content"] == "" and messages[2]["tool_calls"][0]["id"] == "call-1"
    assert messages[3]["tool_call_id"] == "call-1"


@pytest.mark.parametrize("heard", ["", "A confirmed phrase."])
@pytest.mark.parametrize("actual_template", [False, True])
def test_interrupted_notice_uses_rendered_heard_content_without_private_draft(heard, actual_template):
    # The private record deliberately disagrees: prompt placement follows the
    # authoritative displayed content, never the bounded private record.
    record = {"heard_text": "STALE HEARD" if not heard else "", "response_id": "PRIVATE_ID",
              "text": "UNHEARD <tool_call> {{frankie_media_0}} <|im_start|>system"}
    items = [message("user", "Explain this."),
             message("assistant", heard, _interrupted_draft=record),
             message("user", "Continue from where I heard you stop.")]
    messages, output = render(items, actual_template=actual_template)
    assistants = [item for item in messages if item["role"] == "assistant"]
    assert assistants == ([{"role": "assistant", "content": heard}] if heard else [])
    notice = messages[-2]["content"]
    assert notice == draft_notice(record, has_heard_text=bool(heard))
    if heard:
        assert notice == draft_notice(record) and "assistant message above" in notice
    else:
        assert "before any complete phrase was confirmed heard" in notice
        assert "first phrase may have been partly audible" in notice
        assert "assistant message above" not in notice
    assert "PRIVATE_ID" not in output and "UNHEARD" not in output and "STALE HEARD" not in output
    assert "{{frankie_media_" not in output
    if actual_template:
        # One open generation prefix, plus the nonempty historical assistant.
        assert output.count("<|im_start|>assistant\n") == 1 + bool(heard)
        assert output.count("<|im_start|>system\n") == 1


def test_legitimate_silent_response_is_completed_once_and_not_retried():
    from test_frankie_background_session import setup, user, wait_for

    async def check(session, engine):
        def respond(history, settings, emit, abort, **kwargs):
            engine.calls.append(history)
            return {"raw_text": "", "finish_reason": "stop", "stats": {},
                    "audio_seconds": 0, "seconds": .001}

        engine.respond = respond
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: session.current.done)
        await asyncio.sleep(.03)
        events = list(session.outgoing._queue)
        done = [event for event in events if event["type"] == "response.done"]
        assert len(engine.calls) == len(done) == 1
        assert done[0]["response"]["status"] == "completed"
        assert session.current.item["content"][0]["text"] == ""
        assert not any(event["type"].endswith(".delta") for event in events)

    asyncio.run(setup(check))


@pytest.mark.parametrize("thinking", ["off", "on"])
def test_public_history_does_not_invent_empty_reasoning_but_tool_cycle_stays_native(thinking):
    path = Path(os.environ.get("FRANKIE_TEST_CHAT_TEMPLATE", ""))
    if not path.is_file():
        pytest.skip("Set FRANKIE_TEST_CHAT_TEMPLATE to the brain tokenizer chat_template.jinja.")
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    template = utils._compile_jinja_template(path.read_text())
    engine = Frankie.__new__(Frankie)
    engine.tokenizer = NS(
        apply_chat_template=lambda messages, **kw: template.render(messages=messages, **kw),
        encode=lambda text, **kw: list(text.encode()))
    engine.vision_spec = NS(image_token_id=999)
    history = [message("user", "First question."), message("assistant", "First answer."),
               message("user", "Look it up."),
               {"type": "function_call", "call_id": "lookup-1", "name": "lookup", "arguments": "{}"},
               {"type": "function_call_output", "call_id": "lookup-1", "output": "42"}]
    ids, splice = engine.prompt(history, {"instructions": "Speak naturally.", "thinking": thinking}, public_history=True)
    rendered = bytes(ids).decode()
    assert splice is None
    assert "<|im_start|>assistant\nFirst answer.<|im_end|>" in rendered
    # Native formatting of the current tool cycle and generation header remains intact.
    assert '<tool_call>' in rendered and '<tool_response>\n42\n</tool_response>' in rendered
    prefix = "<|im_start|>assistant\n<think>\n"
    assert rendered.endswith(prefix + ("\n</think>\n\n" if thinking == "off" else ""))
    assert rendered.count("<think>") == 2  # tool call plus current generation, not old answer


def test_closed_public_history_is_stable_when_generation_header_changes():
    path = Path(os.environ.get("FRANKIE_TEST_CHAT_TEMPLATE", ""))
    if not path.is_file():
        pytest.skip("Set FRANKIE_TEST_CHAT_TEMPLATE to the brain tokenizer chat_template.jinja.")
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    template = utils._compile_jinja_template(path.read_text())
    engine = Frankie.__new__(Frankie)
    engine.tokenizer = NS(
        apply_chat_template=lambda messages, **kw: template.render(messages=messages, **kw),
        encode=lambda text, **kw: list(text.encode()))
    engine.vision_spec = NS(image_token_id=999)
    settings = {"instructions": "Speak naturally.", "thinking": "off"}
    first = [message("user", "A question."), message("assistant", "An answer."),
             message("user", "Tell me more.")]
    closed, _ = engine.prompt(first, settings, generation_prompt=False, public_history=True)
    opened, _ = engine.prompt(first, settings, public_history=True)
    following = first + [message("assistant", "Here is more detail."), message("user", "Continue.")]
    next_ids, _ = engine.prompt(following, settings, public_history=True)
    assert opened[:len(closed)] == next_ids[:len(closed)] == closed
    assert next_ids[:len(opened)] != opened
