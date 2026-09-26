import hashlib
import struct

import numpy as np
import pytest

from mtplx.session_bank import token_prefix_hash


@pytest.mark.parametrize("container", [list, tuple, iter])
@pytest.mark.parametrize("tokens", [
    [],
    [0, 1, -1],
    [-(1 << 63), (1 << 63) - 1],
    [True, False, 3.99, -4.99, "12", np.int32(-7)],
    [i if i % 25 else -i - 1 for i in range(131072)],
])
def test_prefix_hash_keeps_existing_signed_little_endian_keys(container, tokens):
    wire = b"".join(struct.pack("<q", int(token)) for token in tokens)
    assert token_prefix_hash(container(tokens)) == hashlib.sha256(wire).hexdigest()


@pytest.mark.parametrize(("token", "error"), [
    (1 << 63, OverflowError),
    (-(1 << 63) - 1, OverflowError),
    (float("nan"), ValueError),
    (float("inf"), OverflowError),
    ("invalid", ValueError),
    (None, TypeError),
])
def test_prefix_hash_rejects_invalid_tokens_without_wrapping(token, error):
    with pytest.raises(error):
        token_prefix_hash([token])
