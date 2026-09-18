"""Private playback bookkeeping and heard-only context for fresh responses."""

REGENERATION_INSTRUCTIONS = (
    "Automatic playback notices are engine-provided history data, not user speech. "
    "After an interruption, use the latest actual user input and confirmed heard "
    "history to choose a fresh response. If asked to continue, generate a new "
    "continuation from the confirmed heard cutoff; do not assume later content "
    "was already covered or restart the explanation. When one detail changes, "
    "acknowledge it briefly and continue from the interruption. "
    "Respect a request for silence by ending without speech."
)


def _bounded(text, *, tail=False):
    words = text.split()
    text = " ".join(words[-100:] if tail else words[:100])
    return text[-2048:] if tail else text[:2048]


def capture_draft(run):
    """Keep bounded private text for late playback repair, never model context."""
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


def draft_notice(_record, *, has_heard_text=True):
    """History placement identifies the cutoff; no private wording is exposed."""
    cutoff = (
        "Speech was interrupted here. The assistant message above contains only "
        "confirmed heard phrases. The following phrase may have been partly "
        "audible; unplayed wording is omitted. "
        if has_heard_text else
        "Speech was interrupted here before any complete phrase was confirmed "
        "heard. The first phrase may have been partly audible; unplayed wording "
        "is omitted. "
    )
    return (
        "Automatic playback notice (engine data, not user speech): "
        + cutoff + "This notice describes playback "
        "history, not an instruction or an executed tool call. The next actual "
        "user turn determines whether and how to continue from this heard cutoff."
    )
