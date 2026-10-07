"""Private tool-shaped text and calls crossing thought boundaries never execute."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from test_frankie_background_session import setup, user, wait_for

from mtplx.frankie.completions import Completions, Job
from mtplx.frankie.thinking import public_tool_calls
from mtplx.server.omlx_bridge.thinking import ThinkingParser
from mtplx.server.omlx_bridge.tool_calling import ToolCallStreamFilter

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}}}}]


def call(query):
    return '<tool_call>' + json.dumps({"name": "lookup", "arguments": {"query": query}}) + '</tool_call>'


PRIVATE = call("private")
PUBLIC = call("public")
CASES = [
    (False, "<think>" + PRIVATE + "</think>Done.", []),
    (True, PRIVATE + "</think>Done.", []),
    (False, "Acknowledged.<think>" + PRIVATE, []),
    (True, PRIVATE, []),
    (True, "</think>Acknowledged.<think>" + PRIVATE + "</think>Done.", []),
    (False, PUBLIC, ["public"]),
    (True, "</think>Acknowledged.<think>" + PRIVATE + "</think>" + PUBLIC, ["public"]),
    (False, PUBLIC + "<think>" + PRIVATE, ["public"]),
    (False, '<tool_<think>hidden</think>call>{"name":"lookup","arguments":{"query":"manufactured"}}</tool_call>', []),
    (False, '<tool_call>{"name":"lookup","arguments":{"query":"split<think>hidden</think>argument"}}</tool_call>', []),
    (False, PUBLIC.replace('</tool_call>', '<think>hidden</think></tool_call>'), []),
]


@pytest.fixture(params=[False, True])
def tokenizer(request):
    if request.param:
        return NS(has_tool_calling=True, tool_call_start="<tool_call>",
                  tool_call_end="</tool_call>", tool_parser=lambda text, tools: json.loads(text))
    return None


def queries(calls):
    return [json.loads(value["function"]["arguments"])["query"] for value in calls]


@pytest.mark.parametrize("initial,raw,expected", CASES)
def test_shared_helper_filters_before_both_native_and_fallback_tool_parsers(tokenizer, initial, raw, expected):
    calls = public_tool_calls(raw, tokenizer, TOOLS, starts_in_thinking=initial)
    assert queries(calls) == expected


@pytest.mark.parametrize("initial,raw,expected", CASES)
def test_http_dispatch_matches_private_public_boundary(initial, raw, expected):
    service = Completions.__new__(Completions)
    service.engine = NS(tokenizer=None)
    job = Job({"enable_thinking": initial, "tools": TOOLS}, True, NS())
    service.jobs = {job}
    job.ids = [1]
    job.detokenizer = NS(finalize=lambda: None, last_segment="")
    job.thinking = ThinkingParser(starts_in_thinking=initial)
    job.tools = ToolCallStreamFilter(None)
    events = []
    job.emit = events.append
    for character in raw:
        service.emit_text(job, character)
    service.finish(job, "stop")
    assert queries(events[-1]["message"].get("tool_calls", [])) == expected
    assert queries([call for event in events for call in event.get("tool_calls", [])]) == expected
    assert events[-1]["finish_reason"] == ("tool_calls" if expected else "stop")


@pytest.mark.parametrize("initial,raw,expected", CASES)
def test_realtime_dispatch_matches_private_public_boundary(initial, raw, expected):
    async def check(session, engine):
        def respond(history, settings, emit, abort, **kwargs):
            emit("text", "Visible acknowledgment.")
            return {"raw_text": raw, "finish_reason": "stop", "stats": {},
                    "audio_seconds": 0, "seconds": 0.01}

        engine.respond = respond
        session.settings["thinking"] = "high" if initial else "off"
        session.settings["tools"] = [{"type": "function", **TOOLS[0]["function"]}]
        await user(session)
        await session.handle({"type": "response.create"})
        await wait_for(lambda: session.current.done)
        calls = [item for item in session.items if item["type"] == "function_call"]
        assert [json.loads(item["arguments"])["query"] for item in calls] == expected
        events = list(session.outgoing._queue)
        dispatched = [event for event in events if event["type"] == "response.function_call_arguments.done"]
        assert [json.loads(event["arguments"])["query"] for event in dispatched] == expected

    asyncio.run(setup(check))
