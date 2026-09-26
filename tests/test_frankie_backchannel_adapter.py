"""The fast gate must spend ear work only, on the existing inference owner."""
import asyncio
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_frankie_prefix_adapter import fixture, snapshot


@pytest.mark.parametrize("mtp", [0, 2])
def test_fast_prefix_pairs_ear_snapshots_without_brain_warmup_or_cache(mtp):
    async def check(listener, service, session, ear, owner):
        snap = snapshot()
        evidence, stats = await listener.hear_prefix(snap)
        assert evidence.snapshot is snap and evidence.text == "Actually"
        assert evidence.previous_text == "Actually" and evidence.stable(160)
        assert len(ear.calls) == 2 and len(ear.calls[-1][0]) == snap.samples
        assert service.listener_bank is None and not service.jobs
        assert listener.job is None and "completion_tokens" not in stats
    asyncio.run(fixture(check, ["Actually", "Actually"], mtp=mtp, real_service=True))


@pytest.mark.parametrize("text,action", [("Yeah.", "continue"), ("Actually", "yield"), ("", "continue")])
def test_final_short_overlap_uses_ear_once_and_publishes_complete_features(text, action):
    async def check(listener, service, session, ear, owner):
        part = {"type": "input_audio", "_pcm": np.zeros(9600, dtype=np.float32), "_rate": 24000}
        item = {"id": "utterance", "content": [part], "_speech_voiced_ms": 320, "_speech_elapsed_ms": 400}
        decision, observed, stats = await listener.classify_backchannel(item, NS())
        assert decision.action == action and observed.user_text == text
        assert observed.is_final and part["_transcript"] == text and "_rows" in part
        assert len(ear.calls) == 1 and not service.prompts
        assert service.listener_bank is None and not service.jobs
        assert stats["policy"] == "brief_backchannel" and "completion_tokens" not in stats
        await listener.classify_backchannel(item, NS())
        assert len(ear.calls) == 1  # Reuse only immutable completed ear features.
    asyncio.run(fixture(check, [text]))
