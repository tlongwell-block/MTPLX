"""Frankie tool dispatch sees only complete public spans of native reasoning."""

from mtplx.server.omlx_bridge.thinking import ThinkingParser
from mtplx.server.omlx_bridge.tool_calling import parse_tool_calls


def public_tool_calls(raw, tokenizer, tools, *, starts_in_thinking=False):
    parser = ThinkingParser(starts_in_thinking=starts_in_thinking)
    spans = []
    parser.feed(raw, public_spans=spans)
    _, tail = parser.finish()
    if tail:
        spans[-1] += tail
    # Never concatenate across a private block: two otherwise invalid tool
    # fragments must not become an executable envelope or argument string.
    return [call for span in spans if span.strip()
            for call in (parse_tool_calls(span, tokenizer, tools).tool_calls or [])]
