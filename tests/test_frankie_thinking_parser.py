"""Frankie's shared HTTP parser never recovers private thoughts as content."""

from types import SimpleNamespace as NS

import pytest

from mtplx.frankie.completions import Completions, Job
from mtplx.server.omlx_bridge.thinking import ThinkingParser
from mtplx.server.omlx_bridge.tool_calling import ToolCallStreamFilter


@pytest.mark.parametrize("size", [1, 2, 7, 10000])
@pytest.mark.parametrize("initial", [False, True])
def test_multiple_native_blocks_survive_every_marker_fragmentation(size, initial):
    text = ("</think>" if initial else "") + (
        "Let me check.<think>PRIVATE ONE</think> Details found."
        "<think>PRIVATE TWO</think> Here is the answer.")
    parser = ThinkingParser(starts_in_thinking=initial)
    results = [parser.feed(text[offset:offset + size]) for offset in range(0, len(text), size)]
    results.append(parser.finish())
    assert "".join(private for private, _ in results) == "PRIVATE ONEPRIVATE TWO"
    assert "".join(public for _, public in results) == "Let me check. Details found. Here is the answer."
    assert parser.finish() == ("", "")


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("tail", ["PRIVATE", "PRIVATE <", "PRIVATE </thi"])
def test_truncated_private_blocks_remain_private_and_finish_does_not_duplicate(initial, tail):
    text = ("" if initial else "<think>") + tail
    parser = ThinkingParser(starts_in_thinking=initial)
    results = [parser.feed(character) for character in text] + [parser.finish()]
    assert "".join(private for private, _ in results) == tail
    assert "".join(public for _, public in results) == ""
    assert parser.finish() == ("", "")


@pytest.mark.parametrize("text", ["Hello, there.", "The condition is x < 3.", "A literal <unknown> tag."])
def test_think_off_direct_text_is_unchanged(text):
    parser = ThinkingParser()
    results = [parser.feed(character) for character in text] + [parser.finish()]
    assert "".join(private for private, _ in results) == ""
    assert "".join(public for _, public in results) == text


@pytest.mark.parametrize("reason", ["stop", "length"])
@pytest.mark.parametrize("public_before", ["", "Let me check."])
def test_http_finish_keeps_private_tail_out_of_stream_and_final_content(reason, public_before):
    service = Completions.__new__(Completions)
    service.engine = NS(tokenizer=None)
    job = Job({}, True, NS())
    service.jobs = {job}
    job.ids = [1]
    job.detokenizer = NS(finalize=lambda: None, last_segment="")
    job.thinking = ThinkingParser(starts_in_thinking=True)
    job.tools = ToolCallStreamFilter(None)
    events = []
    job.emit = events.append
    text = "</think>" + public_before + "<think>" if public_before else ""
    text += "PRIVATE UNFINISHED </thi"
    for character in text:
        service.emit_text(job, character)
    service.finish(job, reason)
    assert "".join(event.get("content", "") for event in events) == public_before
    assert "".join(event.get("reasoning_content", "") for event in events) == "PRIVATE UNFINISHED </thi"
    assert events[-1]["message"] == {
        "content": public_before, "reasoning_content": "PRIVATE UNFINISHED </thi"}
    assert events[-1]["finish_reason"] == reason and job not in service.jobs
