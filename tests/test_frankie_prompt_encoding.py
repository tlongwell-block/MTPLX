"""Exact segment memoization must not infer token prefixes or retain unbounded text."""

from types import SimpleNamespace

import pytest

from mtplx.frankie.prompt_encoding import PromptEncoder


def encoder(**kwargs):
    calls = []

    def encode(text, *, add_special_tokens):
        assert not add_special_tokens
        calls.append(text)
        # Different token boundaries for the full string vs separate pieces.
        return [1000] if text == "ab" else list(text.encode())

    return PromptEncoder(SimpleNamespace(encode=encode), **kwargs), calls


def test_exact_text_reuse_keeps_unicode_and_token_boundaries():
    encode, calls = encoder()
    for text in ("a", "ab", "b", "日本語 🐝", "<|im_start|>assistant\n"):
        expected = [1000] if text == "ab" else list(text.encode())
        first = encode(text)
        assert list(first) == expected
        assert list(encode(text)) == expected
        assert calls.count(text) == 1
        with pytest.raises(TypeError):
            first[0] = 999
    assert list(encode("aB")) != list(encode("ab"))


def test_lru_eviction_and_clear_do_not_change_outputs():
    encode, calls = encoder(max_entries=2)
    for text in ("one", "two", "one", "three", "one", "two"):
        assert list(encode(text)) == list(text.encode())
    assert calls == ["one", "two", "three", "two"]
    encode.clear()
    assert not encode.entries and encode.nbytes == 0
    assert list(encode("two")) == list(b"two")
    assert calls.count("two") == 3


@pytest.mark.parametrize("max_bytes", [0, 1024, 4096])
def test_byte_bound_and_oversize_bypass(max_bytes):
    encode, calls = encoder(max_bytes=max_bytes)
    for i in range(100):
        text = "🐝" * i
        assert list(encode(text)) == list(text.encode())
        assert encode.nbytes <= max_bytes
    oversized = "garden " * 3000
    for _ in range(2):
        assert list(encode(oversized)) == list(oversized.encode())
    assert calls.count(oversized) == 2
    assert oversized not in encode.entries


def test_encoders_do_not_share_tokenizer_or_private_text():
    first, first_calls = encoder()
    second = PromptEncoder(SimpleNamespace(encode=lambda text, **kwargs: [4321]))
    assert list(first("ab")) == [1000]
    assert list(second("ab")) == [4321]
    second.clear()
    assert list(first("ab")) == [1000] and len(first_calls) == 1


def test_long_audio_history_scan_keeps_the_expensive_initial_segment():
    encode, calls = encoder()
    # Each native audio span separates another rendered text segment. A small
    # entry-count cap would evict every segment during each sequential scan.
    segments = ["Reference notes. " * 4000] + [
        f"Assistant answer before audio turn {i}." for i in range(256)
    ]
    expected = [list(encode(text)) for text in segments]
    for _ in range(3):
        assert [list(encode(text)) for text in segments] == expected
    assert calls == segments
    assert encode.nbytes <= encode.max_bytes
