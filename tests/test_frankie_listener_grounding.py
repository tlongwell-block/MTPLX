"""Repair gets the controller's lexical evidence without replacing neural audio."""
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

from mtplx.frankie.engine import Frankie


@pytest.mark.parametrize('grounded', [False, True])
def test_grounding_preserves_media_and_leaves_ordinary_turns_unchanged(grounded):
    engine = Frankie.__new__(Frankie)
    seen = []
    transcript = 'Actually, Thursday.'
    rows = mx.ones((2, 4))
    part = {'type': 'input_audio', '_rows': rows, '_transcript': transcript}
    if grounded:
        part['_listener_transcript'] = transcript

    def template(messages, **kwargs):
        seen.extend(messages)
        return '\n'.join(m['content'] for m in messages)

    engine.tokenizer = NS(apply_chat_template=template, encode=lambda text, **kwargs: list(text.encode()))
    engine.vision_spec = NS(image_token_id=999)
    ids, splice = engine.prompt([{'type': 'message', 'role': 'user', 'content': [part]}],
                               {'instructions': 'Speak naturally.', 'thinking': 'off'})
    assert ids.count(999) == len(rows)
    assert splice is not None and part['_rows'] is rows
    assert ('Speech transcript (may contain errors): ' + transcript in seen[-1]['content']) is grounded
    assert '{{frankie_media_0}}' in seen[-1]['content']
