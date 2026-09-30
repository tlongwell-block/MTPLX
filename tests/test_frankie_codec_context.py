from types import SimpleNamespace as NS

import pytest

from mtplx.frankie.codec_context import CodecContext


class Decoder:
    def __init__(self):
        self._transformer_cache = None
        self.convolution = NS(_buffer=None, _overflow=None)

    def named_modules(self):
        return [("convolution", self.convolution)]

    def reset_streaming_state(self):
        self._transformer_cache = None
        self.convolution._buffer = self.convolution._overflow = None


def test_convolution_survives_phrase_boundary_but_not_response_boundary():
    decoder = Decoder()
    context = CodecContext(decoder, "convolution")
    reset = decoder.reset_streaming_state
    with context.response() as owner:
        with context.phrase(owner):
            decoder.convolution._buffer = NS(nbytes=4)
            decoder._transformer_cache = [NS(offset=3, keys=NS(nbytes=4))]
        buffer = decoder.convolution._buffer
        decoder.reset_streaming_state()
        assert decoder._transformer_cache is None
        assert decoder.convolution._buffer is buffer
        with pytest.raises(RuntimeError, match="owned"):
            with context.response():
                pytest.fail("Concurrent response obtained the same decoder")
        with pytest.raises(RuntimeError, match="does not own"):
            context.clear(object(), "foreign")
    assert decoder.reset_streaming_state == reset
    assert decoder.convolution._buffer is None and context._owner is None
    with context.response() as owner:
        assert decoder.convolution._buffer is None


@pytest.mark.parametrize("failure", ["cancel", "overflow"])
def test_failed_phrase_releases_state_and_owner(failure):
    decoder = Decoder()
    context = CodecContext(decoder, "convolution", max_bytes=8)
    exception = GeneratorExit if failure == "cancel" else RuntimeError
    with pytest.raises(exception):
        with context.response() as owner:
            with context.phrase(owner):
                decoder.convolution._buffer = NS(nbytes=16)
                if failure == "cancel":
                    raise GeneratorExit
    assert context._owner is None and decoder.convolution._buffer is None
    with context.response():
        pass


def test_unqualified_full_attention_continuation_is_not_exposed():
    with pytest.raises(ValueError, match="mode"):
        CodecContext(Decoder(), "full")
