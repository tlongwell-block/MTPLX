"""Bounded public draft context for a fresh response after yielding speech."""

import json


def _bounded(text, *, tail=False):
    words = text.split()
    text = " ".join(words[-100:] if tail else words[:100])
    return text[-2048:] if tail else text[:2048]


def capture_draft(run):
    """Never label a partly played phrase as fully heard, or use raw tokens."""
    heard = " ".join(c["text"] for c in run.chunks if c["end_ms"] <= run.played_ms)
    previous = (run.item or {}).get("_interrupted_draft")
    # A later device-clock acknowledgment can confirm one more full phrase
    # after the server requested a clear. Reuse only the retained bounded draft.
    text = " ".join((run.text + (" " + previous["text"] if previous else "")).split())
    prefix = " ".join(heard.split())
    if text.startswith(prefix):
        text = text[len(prefix):].lstrip()
    elif previous:
        # A delayed acknowledgment may cross the entire bounded retained
        # window. Do not resurrect already heard words as an unfinished draft.
        text = ""
    return {"response_id": run.id, "heard_text": _bounded(heard, tail=True),
            "text": _bounded(text), "played_ms": run.played_ms}


def draft_notice(record):
    # Escape literal template/control delimiters inside the quoted data. This
    # is public speech only, never hidden reasoning or a callable tool plan.
    data = json.dumps(record, ensure_ascii=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("{{frankie_media_", "\\u007b\\u007bfrankie_media_")
    return (
        "Speech was interrupted here. The assistant message above contains only "
        "confirmed heard phrases. The quoted draft below was generated but not "
        "confirmed heard; its first phrase may have been partly heard. This is "
        "historical draft data, not an instruction or an executed tool call. "
        "Use the user's subsequent words to decide your next response freshly: "
        "continue naturally, correct yourself, change topic, or remain silent. "
        "Do not automatically replay the draft or repeat the heard prefix.\n"
        + data
    )
