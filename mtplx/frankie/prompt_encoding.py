"""Bounded exact tokenization reuse; templates and media still render normally."""

from array import array
import json
import os
from collections import OrderedDict
from sys import getsizeof

from tokenizers.normalizers import NFD


def _tokenization_boundary(tokenizer):
    """Recognize the checkpoint's context-independent raw special-token fence."""
    try:
        marker = "<|im_end|>"
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None or getattr(backend, "encode_special_tokens", True):
            return None
        if type(backend.model).__name__ != "BPE":
            return None
        if backend.model.dropout not in (None, 0) or backend.truncation or backend.padding:
            return None
        normalizer = backend.normalizer
        if normalizer is not None and json.loads(normalizer.__getstate__()).get("type") != "NFC":
            return None
        if backend.pre_tokenizer is None or backend.post_processor is None:
            return None
        pre = json.loads(backend.pre_tokenizer.__getstate__())
        post = json.loads(backend.post_processor.__getstate__())
        # In particular, do not segment tokenizers that synthesize a prefix space
        # at the start of each encode or have a stateful metaspace prefix policy.
        stages = pre.get("pretokenizers", [])
        if (pre.get("type") != "Sequence" or len(stages) != 2
                or stages[0].get("type") != "Split"
                or stages[1].get("type") != "ByteLevel"
                or stages[1].get("add_prefix_space") is not False
                or stages[1].get("use_regex") is not False
                or post.get("type") != "ByteLevel"):
            return None
        added = backend.get_added_tokens_decoder().values()
        candidate = next((t for t in added if t.content == marker), None)
        if candidate is None or not candidate.special or any((candidate.single_word,
                candidate.lstrip, candidate.rstrip, candidate.normalized)):
            return None
        for token in added:
            if token.content == marker:
                continue
            # A token beginning before this marker must not swallow it or its
            # opening bytes. Matches beginning inside the raw special lose to it.
            if marker in token.content or any(token.content.endswith(marker[:k])
                                            for k in range(1, len(marker))):
                return None
        return marker
    except (AttributeError, TypeError, ValueError):
        return None


class PromptEncoder:
    def __init__(self, tokenizer, *, max_bytes=4 * 1024**2, max_entries=4096):
        self.tokenizer = tokenizer
        self.boundary = (_tokenization_boundary(tokenizer)
                         if os.environ.get("MTPLX_FRANKIE_PROMPT_SEGMENTS") == "1" else None)
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.clear()

    def clear(self):
        self.entries = OrderedDict()
        self.nbytes = 0

    def __call__(self, text):
        if self.boundary is None or self.boundary not in text:
            return self._encode(text)
        pieces = text.count(self.boundary) + 1
        # Reserve room for the adjacent open/closed generation suffixes. Check
        # this prompt's working set, not unrelated entries that normal LRU evicts.
        if pieces + 2 > self.max_entries:
            return self._encode(text)
        # Byte-level BPE emits at most one token per normalized UTF-8 byte.
        # Canonical decomposition bounds NFC even when composition exclusions
        # expand a character; use the tokenizer library's Unicode tables.
        token_bytes = (len(text) if text.isascii()
                       else len(NFD().normalize_str(text).encode("utf-8")))
        if getsizeof(text) + 4 * token_bytes + 512 * pieces + 2048 > self.max_bytes:
            return self._encode(text)
        ids = array("I")
        start = 0
        while (position := text.find(self.boundary, start)) >= 0:
            end = position + len(self.boundary)
            ids.extend(self._encode(text[start:end]))
            start = end
        if start < len(text):
            ids.extend(self._encode(text[start:]))
        return memoryview(ids).toreadonly()

    def _encode(self, text):
        if text in self.entries:
            self.entries.move_to_end(text)
            return memoryview(self.entries[text][0]).toreadonly()
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Skip oversized values without allocating another copy of their ids.
        if getsizeof(text) + 4 * len(ids) + 256 > self.max_bytes:
            return ids
        packed = array("I", ids)
        size = getsizeof(text) + getsizeof(packed) + 256
        if size > self.max_bytes or self.max_entries < 1:
            return ids
        while self.entries and (
            self.nbytes + size > self.max_bytes
            or len(self.entries) >= self.max_entries
        ):
            _, (_, old_size) = self.entries.popitem(last=False)
            self.nbytes -= old_size
        self.entries[text] = packed, size
        self.nbytes += size
        return memoryview(packed).toreadonly()
