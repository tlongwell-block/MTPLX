"""Request-local text buffers with one immutable BPE table per brain."""

import copy


def new_detokenizer(owner):
    # TokenizerWrapper.detokenizer constructs a fresh vocabulary map on
    # every access. Stock BPE.reset replaces all mutable request state;
    # its tokenmap/byte decoder are read-only and safe to share. Other
    # detokenizer implementations retain their existing construction path.
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer
    template = getattr(owner, "detokenizer_template", None)
    if template is None:
        template = owner.tokenizer.detokenizer
        if type(template) is BPEStreamingDetokenizer:
            owner.detokenizer_template = template
    result = copy.copy(template)
    result.reset()
    return result
