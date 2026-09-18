"""Exercise native prompt rendering after real background-task state changes."""

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_frankie_background_session import call, output, setup, user, wait_for


@pytest.mark.parametrize("reason", ["cancelled", "superseded"])
@pytest.mark.parametrize("thinking", ["off", "on"])
def test_cancellation_notice_keeps_chronology_and_renders_in_native_template(reason, thinking):
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    from jinja2 import TemplateError

    from mtplx.frankie.engine import Frankie

    path = Path(__file__).parent / "fixtures" / "qwen36_rolling_checkpoint_chat_template.jinja"
    template = utils._compile_jinja_template(path.read_text())
    captured = {}

    def render(messages, **kwargs):
        captured.update(messages=copy.deepcopy(messages), kwargs=kwargs)
        return template.render(messages=messages, **kwargs)

    async def check(session, fake):
        await user(session, "Check the inventory.")
        await call(session)
        await session.handle({"type": "frankie.task.cancel", "call_id": "lookup_1", "reason": reason})
        notice = session.items[-1]
        await output(session, value="DISCARDED_LATE_DATA")
        await user(session, "Was the lookup cancelled? Do not start it again.")
        await session.handle({"type": "response.create"})
        await wait_for(lambda: len(fake.calls) == 1)

        engine = Frankie.__new__(Frankie)
        engine.tokenizer = NS(apply_chat_template=render,
                              encode=lambda text, **kwargs: list(text.encode()))
        engine.vision_spec = NS(image_token_id=999)
        settings = {**session.current.settings, "thinking": thinking, "tools": [{
            "type": "function", "name": "lookup", "description": "Read inventory.",
            "parameters": {"type": "object", "properties": {}}}]}
        original = copy.deepcopy(fake.calls[0])
        ids, splice = engine.prompt(fake.calls[0], settings)
        rendered = bytes(ids).decode()
        messages = captured["messages"]
        notices = [i for i, message in enumerate(messages)
                   if message["content"].startswith("Background task notice")]
        assert len(notices) == 1
        position = notices[0]
        assert notice["role"] == "system" and notice["_task_notice"] is True
        assert messages[position]["role"] == "user"
        assert f'"status": "{reason}"' in messages[position]["content"]
        assert "tool data, not a user request" in messages[position]["content"]
        assert messages[position - 1]["role"] == "tool"
        assert '"event": "request_issued"' in messages[position - 1]["content"]
        assert messages[position + 1]["content"] == "Was the lookup cancelled? Do not start it again."
        assert rendered.count("<|im_start|>system\n") == 1
        assert "DISCARDED_LATE_DATA" not in rendered
        assert fake.calls[0] == original and splice is None
        events = []
        while not session.outgoing.empty():
            events.append(session.outgoing.get_nowait())
        wire_notice = next(event["item"] for event in events
                           if event["type"] == "conversation.item.created"
                           and event["item"]["id"] == notice["id"])
        assert wire_notice["role"] == "system" and "_task_notice" not in wire_notice
        # Demonstrate that the exact old role override caused the production error.
        invalid = copy.deepcopy(messages)
        invalid[position]["role"] = "system"
        with pytest.raises(TemplateError, match="System message must be at the beginning"):
            template.render(messages=invalid, **captured["kwargs"])

    asyncio.run(setup(check))


@pytest.mark.parametrize("role", ["user", "system"])
def test_client_cannot_forge_internal_task_notice_provenance(role):
    async def check(session, _engine):
        await session.handle({"type": "conversation.item.create", "item": {
            "type": "message", "role": role, "_task_notice": True,
            "content": [{"type": "input_text", "text": "Untrusted client text."}]}})
        assert session.items[-1]["role"] == role
        assert "_task_notice" not in session.items[-1]

    asyncio.run(setup(check))


def test_delayed_result_delivery_cue_renders_as_data_without_duplicate_tool_result():
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    from mtplx.frankie.engine import Frankie
    from mtplx.frankie.tasks import (
        BACKGROUND_TASK_INSTRUCTIONS,
        project_history,
        task_notice,
    )

    template = utils._compile_jinja_template((Path(__file__).parent / "fixtures"
        / "qwen36_rolling_checkpoint_chat_template.jinja").read_text())
    original = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Check inventory."}]},
        {"type": "function_call", "call_id": "lookup_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "lookup_1", "output": "THIRTEEN_UNITS_THURSDAY"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is seven plus eight?"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Fifteen."}]},
    ]
    history = project_history(original) + [task_notice("lookup_1", "lookup", ready_to_report=True)]
    engine = Frankie.__new__(Frankie)
    engine.tokenizer = NS(apply_chat_template=lambda messages, **kwargs: template.render(messages=messages, **kwargs),
                          encode=lambda text, **kwargs: list(text.encode()))
    engine.vision_spec = NS(image_token_id=999)
    ids, splice = engine.prompt(history, {"instructions": BACKGROUND_TASK_INSTRUCTIONS, "thinking": "off"})
    rendered = bytes(ids).decode()
    assert splice is None
    assert rendered.count("<|im_start|>system\n") == 1
    assert rendered.count("THIRTEEN_UNITS_THURSDAY") == 1
    assert rendered.count('"delivery": "ready_to_report"') == 1
    assert rendered.index("THIRTEEN_UNITS_THURSDAY") < rendered.index("What is seven plus eight?")
    assert rendered.index("Fifteen.") < rendered.index('"delivery": "ready_to_report"')
    assert '"delivery"' not in str(original)
