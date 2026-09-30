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


def chat_tokenizer(*, prefix_space=False):
    from tokenizers import AddedToken, Regex, Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

    backend = Tokenizer(models.BPE())
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\s+|[^\s]+"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=prefix_space, use_regex=False),
    ])
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    backend.train_from_iterator(
        ["Hello world. A café. Useful tool result. 中文"],
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    backend.add_special_tokens([AddedToken("<|im_end|>", special=True, normalized=False)])
    calls = []

    def encode(text, **kwargs):
        calls.append(text)
        return backend.encode(text, **kwargs).ids

    return SimpleNamespace(backend_tokenizer=backend, encode=encode), calls


def test_special_fence_reuses_closed_messages_with_exact_unicode_ids(monkeypatch):
    monkeypatch.setenv("MTPLX_FRANKIE_PROMPT_SEGMENTS", "1")
    tokenizer, calls = chat_tokenizer()
    encode = PromptEncoder(tokenizer)
    assert encode.boundary == "<|im_end|>"
    previous = "user\ne\u0301 \u0344 中文🙂<|im_end|>assistant\n"
    for text in (previous, previous + "Hello.<|im_end|>user\nNext<|im_end|>assistant\n"):
        expected = tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids
        assert list(encode(text)) == expected
    assert calls.count("user\ne\u0301 \u0344 中文🙂<|im_end|>") == 1
    assert encode.nbytes <= encode.max_bytes


@pytest.mark.parametrize("limits", [{"max_entries": 3}, {"max_bytes": 64}])
def test_segment_budget_falls_back_before_partial_encoding(monkeypatch, limits):
    monkeypatch.setenv("MTPLX_FRANKIE_PROMPT_SEGMENTS", "1")
    tokenizer, calls = chat_tokenizer()
    encode = PromptEncoder(tokenizer, **limits)
    text = "Hello<|im_end|>world<|im_end|>assistant\n"
    expected = tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids
    assert list(encode(text)) == expected and calls == [text]
    assert encode.nbytes <= encode.max_bytes and len(encode.entries) <= encode.max_entries


def test_context_sensitive_prefix_space_disables_segmentation(monkeypatch):
    monkeypatch.setenv("MTPLX_FRANKIE_PROMPT_SEGMENTS", "1")
    tokenizer, calls = chat_tokenizer(prefix_space=True)
    encode = PromptEncoder(tokenizer)
    assert encode.boundary is None
    text = "Hello<|im_end|>world"
    assert list(encode(text)) == tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids
    assert calls == [text]
