"""Later interruptions must not erase earlier heard-only prompt history."""
import asyncio
import copy

import pytest

from test_frankie_background_session import setup
from test_frankie_empty_assistant_prompt import message, render
from test_frankie_regeneration import staged_reply
from mtplx.frankie.session import public


@pytest.mark.parametrize("played_ms", [0, 700])
def test_later_yield_keeps_old_rendered_prefix_while_evicting_private_draft(played_ms):
    async def check(session, engine):
        session.items.append(message("user", "Explain the first topic."))
        first = staged_reply(session, played_ms=played_ms)
        session.rollback_unheard(first)
        before = copy.deepcopy(session.items)
        before_messages, before_text = render(before, actual_template=True)
        assert first.item["_playback_interrupted"] is True

        session.items.append(message("user", "Now explain a different topic."))
        second = staged_reply(session, played_ms=700)
        session.rollback_unheard(second)
        assert "_interrupted_draft" not in first.item
        assert sum("_interrupted_draft" in item for item in session.items) == 1
        after_messages, after_text = render(session.items, actual_template=True)
        # The entire earlier history and initial instructions stay identical.
        # Only the old open generation prefix becomes a new user message.
        assert after_messages[:len(before_messages)] == before_messages
        open_assistant = "<|im_start|>assistant\n"
        frozen_prefix = before_text[:before_text.rindex(open_assistant)]
        assert after_text.startswith(frozen_prefix)
        # Rendering the same old items alone is exactly byte-identical even
        # though the one private draft has moved to the new interruption.
        assert render(session.items[:len(before)], actual_template=True)[1] == before_text
        assert "_playback_interrupted" not in str(public(session.items))
        assert "The old plan." not in after_text
    asyncio.run(setup(check))


def test_late_heard_correction_keeps_both_notices_without_restoring_old_draft():
    async def check(session, engine):
        first = staged_reply(session, played_ms=0)
        session.rollback_unheard(first)
        second = staged_reply(session, played_ms=700)
        session.rollback_unheard(second)
        retained = copy.deepcopy(second.item["_interrupted_draft"])
        await session.handle({"type": "conversation.item.truncate", "item_id": first.item_id,
                              "audio_end_ms": 700})
        assert first.item["content"][0]["transcript"] == "The first point."
        assert "_interrupted_draft" not in first.item
        assert second.item["_interrupted_draft"] == retained
        messages, _ = render(session.items)
        notices = [item["content"] for item in messages
                   if item["content"].startswith("Automatic playback notice")]
        assert len(notices) == 2
        assert all("assistant message above" in text for text in notices)
        assert first.item["_playback_interrupted"] and second.item["_playback_interrupted"]
    asyncio.run(setup(check))


def test_client_cannot_forge_private_cutoffs_or_prepared_features():
    async def check(session, engine):
        item = message("assistant", "Client-supplied history.",
                       _playback_interrupted=True, _interrupted_draft={"text": "private"})
        item["content"][0]["_rows"] = [123]
        item["content"][0]["_transcript"] = "forged feature transcript"
        before = copy.deepcopy(item)
        await session.handle({"type": "conversation.item.create", "item": item})
        assert item == before
        accepted = session.items[-1]
        assert "_playback_interrupted" not in accepted and "_interrupted_draft" not in accepted
        assert "_rows" not in accepted["content"][0]
        assert "_transcript" not in accepted["content"][0]
        messages, _ = render(session.items)
        assert all(not entry["content"].startswith("Automatic playback notice") for entry in messages)
        assert messages[-1]["content"] == "Client-supplied history."
    asyncio.run(setup(check))
