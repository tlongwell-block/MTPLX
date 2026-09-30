"""Repair gets the controller's lexical evidence without replacing neural audio."""
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

from mtplx.frankie.engine import Frankie


@pytest.mark.parametrize('mode', [None, 'off', 'observe', 'semantic', 'backchannel'])
@pytest.mark.parametrize('grounded', [False, True])
def test_grounding_preserves_media_and_leaves_other_modes_unchanged(grounded, mode):
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
    settings = {'instructions': 'Speak naturally.', 'thinking': 'off'}
    if mode is not None:
        settings['streaming_listener'] = mode
    ids, splice = engine.prompt([{'type': 'message', 'role': 'user', 'content': [part]}], settings)
    assert ids.count(999) == len(rows)
    assert splice is not None and part['_rows'] is rows
    assert ('Speech transcript (may contain errors): ' + transcript in seen[-1]['content']) is (
        grounded or mode == 'backchannel')
    assert '{{frankie_media_0}}' in seen[-1]['content']
    assert seen[-1]['role'] == 'user'
    assert mx.array_equal(splice.embeddings, rows)


@pytest.mark.parametrize('transcript', ['', '   ', 'I meant nine, not twelve.'])
def test_fast_grounding_uses_full_ear_result_once_and_does_not_mutate_history(transcript):
    engine = Frankie.__new__(Frankie)
    rows = mx.ones((2, 4))
    calls, seen = [], []

    def hear(pcm, rate):
        calls.append((pcm, rate))
        return rows, transcript

    def template(messages, **kwargs):
        seen.append(messages)
        return '\n'.join(m['content'] for m in messages)

    engine.audio = NS(hear=hear)
    engine.tokenizer = NS(apply_chat_template=template, encode=lambda text, **kwargs: list(text.encode()))
    engine.vision_spec = NS(image_token_id=999)
    part = {'type': 'input_audio', '_pcm': b'full retained input', '_rate': 24000}
    items = [{'type': 'message', 'role': 'user', 'content': [part]}]
    settings = {'instructions': 'Speak naturally.', 'thinking': 'off', 'streaming_listener': 'backchannel'}
    first_ids, first_splice = engine.prompt(items, settings)
    second_ids, second_splice = engine.prompt(items, settings)
    assert calls == [(b'full retained input', 24000)]
    assert first_ids == second_ids
    assert mx.array_equal(first_splice.embeddings, second_splice.embeddings)
    assert part['_rows'] is rows and part['_transcript'] == transcript
    assert '_listener_transcript' not in part
    assert ('Speech transcript (may contain errors):' in seen[0][-1]['content']) is bool(transcript.strip())
    if transcript.strip():
        assert seen[0][-1]['content'].endswith(transcript)
