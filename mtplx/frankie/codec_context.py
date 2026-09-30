"""Experimental, response-owned native Breeze decoder continuity."""

from contextlib import contextmanager
from threading import Lock


class CodecContext:
    def __init__(self, decoder, mode, *, max_bytes=100_000_000):
        if mode != "convolution":
            raise ValueError("Unknown codec continuation mode.")
        self.decoder, self.max_bytes = decoder, max_bytes
        self._reset = decoder.reset_streaming_state
        self._owner = None
        self._lease = Lock()
        self._in_phrase = False
        self.events = []

    def require_idle(self):
        if self._owner is not None:
            raise RuntimeError("The codec is owned by an active speech response.")

    def require_owner(self, owner):
        if self._owner is None or owner is not self._owner:
            raise RuntimeError("Speech does not own the active codec response.")

    def state(self):
        cache = self.decoder._transformer_cache or []
        kv = sum(getattr(getattr(c, n, None), "nbytes", 0)
                 for c in cache for n in ("keys", "values"))
        convolution = sum(getattr(getattr(m, n, None), "nbytes", 0)
                          for _, m in self.decoder.named_modules()
                          for n in ("_buffer", "_overflow"))
        return {"offset": cache[0].offset if cache else 0,
                "kv_bytes": kv, "convolution_bytes": convolution,
                "total_bytes": kv + convolution}

    def clear(self, owner, reason):
        self.require_owner(owner)
        self.events.append({"reason": reason, **self.state()})
        self._reset()

    def _phrase_reset(self):
        # Only this concrete decoder is rebound, only inside its owned response.
        # Preserve the original reset schedule for attention.
        self.decoder._transformer_cache = None

    @contextmanager
    def response(self):
        if not self._lease.acquire(blocking=False):
            raise RuntimeError("The codec is owned by an active speech response.")
        try:
            self.require_idle()
            self._reset()
            owner = object()
            self._owner = owner
            self.events = []
            self.decoder.reset_streaming_state = self._phrase_reset
            try:
                yield owner
            finally:
                self.decoder.reset_streaming_state = self._reset
                self._owner = None
                self._in_phrase = False
                self._reset()
        finally:
            self._lease.release()

    @contextmanager
    def phrase(self, owner):
        self.require_owner(owner)
        if self._in_phrase:
            raise RuntimeError("A codec phrase is already in progress.")
        self._in_phrase = True
        try:
            yield
            if self.state()["total_bytes"] > self.max_bytes:
                self.clear(owner, "codec_memory_actual")
                raise RuntimeError("Decoder state exceeded the continuation memory bound.")
        except BaseException:
            self.clear(owner, "phrase_not_completed")
            raise
        finally:
            self._in_phrase = False
