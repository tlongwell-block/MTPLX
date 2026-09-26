"""Bounded exact tokenization reuse; templates and media still render normally."""

from array import array
from collections import OrderedDict
from sys import getsizeof


class PromptEncoder:
    def __init__(self, tokenizer, *, max_bytes=4 * 1024**2, max_entries=64):
        self.tokenizer = tokenizer
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.clear()

    def clear(self):
        self.entries = OrderedDict()
        self.nbytes = 0

    def __call__(self, text):
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
