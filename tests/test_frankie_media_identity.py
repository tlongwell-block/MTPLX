"""Media prefix identities stay content-true without rehashing old device rows."""
import hashlib
from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np

from mtplx.frankie.engine import Frankie
from mtplx.vision.splice import vision_bank_key_ids


def test_media_identity_survives_history_replay_but_changes_with_replacement(monkeypatch):
    import mtplx.frankie.engine as module

    engine = Frankie.__new__(Frankie)
    engine.tokenizer = NS(
        apply_chat_template=lambda messages, **kw: '\n'.join(m['content'] for m in messages),
        encode=lambda text, **kw: list(text.encode()),
    )
    engine.vision_spec = NS(image_token_id=999)
    parts = [{'type': kind, '_rows': mx.full((2, 4), value), '_transcript': ''}
             for kind, value in [('input_audio', 2.), ('input_image', 7.)]]
    items = [{'type': 'message', 'role': 'user', 'content': parts}]
    settings = {'instructions': 'Answer.', 'thinking': 'off'}
    real_hash = hashlib.sha256
    hashes = []
    def digest(data):
        hashes.append(bytes(data))
        return real_hash(data)
    monkeypatch.setattr(module.hashlib, 'sha256', digest)
    first_ids, first = engine.prompt(items, settings)
    expected = tuple(int.from_bytes(real_hash(memoryview(np.asarray(p['_rows'].astype(mx.float32)))).digest()[:8], 'little') for p in parts)
    assert first.image_digests == expected
    for generation_prompt in (False, True, False):
        ids, splice = engine.prompt(items, settings, generation_prompt=generation_prompt)
        assert ids == first_ids
        assert mx.array_equal(first.embeddings, splice.embeddings)
        assert vision_bank_key_ids(ids, splice) == vision_bank_key_ids(first_ids, first)
    assert len(hashes) == 2
    # Same marker, shape, and pad count must not reuse the old media's identity.
    parts[0]['_rows'] = mx.full((2, 4), 9.)
    ids, replaced = engine.prompt(items, settings)
    assert len(hashes) == 3 and ids == first_ids
    assert replaced.image_digests[0] != first.image_digests[0]
    assert replaced.image_digests[1] == first.image_digests[1]
    assert vision_bank_key_ids(ids, replaced) != vision_bank_key_ids(first_ids, first)
    # Equal replacement values still qualify for the same content-based prefix.
    parts[0]['_rows'] = mx.full((2, 4), 9.)
    _, equal = engine.prompt(items, settings)
    assert equal.image_digests == replaced.image_digests
    assert len(hashes) == 4
    # Deletion uses the retained part's own identity.
    items[0]['content'] = [parts[1]]
    _, deleted = engine.prompt(items, settings)
    assert deleted.image_digests == (first.image_digests[1],)
    assert deleted.embeddings is parts[1]['_rows']
    assert len(hashes) == 4
