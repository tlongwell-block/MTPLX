# SPDX-License-Identifier: Apache-2.0
"""Thinking parser adapted from oMLX.

Source inspiration: oMLX ``omlx/api/thinking.py`` on origin/main, Apache-2.0.
The streaming parser keeps unfinished native reasoning private. The legacy
non-stream extraction helper below retains its existing recovery behavior.
"""

from __future__ import annotations

import re

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_OPEN_LEN = len(THINK_OPEN)
_CLOSE_LEN = len(THINK_CLOSE)
_THINKING_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINKING_TAIL_PATTERN = re.compile(r"^(.*?)</think>", re.DOTALL)


def extract_thinking(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    thinking_parts: list[str] = []
    remaining = text
    while True:
        match = _THINKING_PATTERN.search(remaining)
        if not match:
            break
        thinking_parts.append(match.group(1))
        remaining = remaining[: match.start()] + remaining[match.end() :]
    if thinking_parts:
        return "\n".join(thinking_parts).strip(), remaining.strip()
    if THINK_CLOSE in text and THINK_OPEN not in text:
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            return match.group(1).strip(), text[match.end() :].strip()
    if THINK_OPEN in text and THINK_CLOSE not in text:
        index = text.index(THINK_OPEN)
        recovered = (text[:index] + text[index + _OPEN_LEN :]).strip()
        return recovered, recovered
    return "", text


class ThinkingParser:
    """Streaming native reasoning; an unfinished block never becomes content."""

    def __init__(self, *, starts_in_thinking: bool = False) -> None:
        self._in_thinking = starts_in_thinking
        self._buffer = ""

    def feed(self, text: str, *, public_spans: list[str] | None = None) -> tuple[str, str]:
        """Optionally collect this feed's public spans without joining across thoughts."""
        if not text:
            return "", ""
        text = self._buffer + text
        self._buffer = ""
        thinking_out: list[str] = []
        content_out: list[str] = []
        span_start = 0
        i = 0
        while i < len(text):
            if text[i] == "<":
                remaining = text[i:]
                if remaining.startswith(THINK_OPEN):
                    if public_spans is not None:
                        public_spans.append("".join(content_out[span_start:]))
                        span_start = len(content_out)
                    self._in_thinking = True
                    i += _OPEN_LEN
                    continue
                if remaining.startswith(THINK_CLOSE):
                    self._in_thinking = False
                    i += _CLOSE_LEN
                    continue
                if self._could_be_tag(remaining):
                    self._buffer = remaining
                    break
            if self._in_thinking:
                thinking_out.append(text[i])
            else:
                content_out.append(text[i])
            i += 1
        thinking = "".join(thinking_out)
        content = "".join(content_out)
        if public_spans is not None:
            public_spans.append("".join(content_out[span_start:]))
        return thinking, content

    def finish(self) -> tuple[str, str]:
        partial = self._buffer
        self._buffer = ""
        if not partial:
            return "", ""
        if self._in_thinking:
            return partial, ""
        return "", partial

    @staticmethod
    def _could_be_tag(text: str) -> bool:
        if len(text) >= _CLOSE_LEN:
            return False
        return THINK_OPEN.startswith(text) or THINK_CLOSE.startswith(text)
