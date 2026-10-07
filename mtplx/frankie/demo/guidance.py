"""Paired classifier-free guidance for Breeze, with both lanes keeping history.

Breeze's native CFG runs an unconditional lane beside the directed one. Here
that lane hears the same voice reference and the same spoken history as the
directed lane, with the instruction left out, so guidance pushes only toward
the delivery. The directed lane owns the model's own context fields; the
unconditional lane keeps its own copy (``shadow``) with its own KV cache.

Weights are never changed. The native generate loop runs unmodified; the one
copied operation is BreezeModel.generate's phrase-completion accounting,
applied to both lanes. ``scale`` 1.0 is the original optimized path exactly.
"""
from collections import deque
from contextlib import contextmanager

FIELDS = ("_speech_cache", "_context_words", "_continuing", "_speech_segments",
          "_chunk_start_rows", "_window_evictions", "_window_evicted_rows")


def empty_state():
    return dict(_speech_cache=None, _context_words=0, _continuing=False,
                _speech_segments=deque(), _chunk_start_rows=0,
                _window_evictions=0, _window_evicted_rows=0)


def capture(model):
    return {name: getattr(model, name) for name in FIELDS}


def activate(model, state):
    for name, value in state.items():
        setattr(model, name, value)


class PairedGuidance:
    """Attach to one BreezeModel; its codec is never replaced.

    ``native_generate`` is the upstream generate loop (without BreezeModel's
    accounting); ``paired_depth`` is ``build_paired_depth(model)``.
    """

    def __init__(self, model, native_generate, paired_depth):
        self.model = model
        self.native_generate = native_generate
        self.paired_depth = paired_depth
        self.original_generate = model.generate
        self.original_prompt = model._prompt_embeddings
        self.original_cache = model.backbone_model.make_cache
        self.original_depth = model._depth_tokens
        self.original_reset = model.reset_speech_context
        self.original_hold = model.hold_speech
        self.shadow = empty_state()
        self.scale = 1.0
        self.active = False
        self.branch_depth = 0
        self.prompt_order = []
        self.cache_order = []
        self.installed = False

    def install(self):
        if self.installed:
            raise RuntimeError("Guidance already installed")
        self.model.generate = self.generate
        self.model._prompt_embeddings = self.prompt
        self.model.backbone_model.make_cache = self.make_cache
        self.model._depth_tokens = self.depth
        self.model.reset_speech_context = self.reset
        self.model.hold_speech = self.hold
        self.installed = True
        return self

    def in_step(self):
        """Whether both lanes hold the same spoken phrases."""
        heard = lambda segments: [words for words, _ in segments]
        return (heard(self.model._speech_segments) == heard(self.shadow["_speech_segments"])
                and self.model._context_words == self.shadow["_context_words"])

    def reset(self):
        self.original_reset()
        if not self.branch_depth:
            self.shadow = empty_state()

    def hold(self):
        """Keep the same newest phrases in both lanes."""
        self.original_hold()
        with self.branch("negative"):
            self.original_hold()

    @contextmanager
    def branch(self, kind):
        public = capture(self.model)
        if kind == "negative":
            activate(self.model, self.shadow)
        self.branch_depth += 1
        try:
            yield
        finally:
            self.branch_depth -= 1
            if kind == "negative":
                self.shadow = capture(self.model)
                activate(self.model, public)

    def prompt(self, *args, **kwargs):
        if not self.active:
            return self.original_prompt(*args, **kwargs)
        expected = ("conditional", "negative")[len(self.prompt_order)] if len(self.prompt_order) < 2 else None
        kind = "negative" if kwargs.get("instruct") is None else "conditional"
        if expected != kind:
            raise RuntimeError("Native CFG prompt order changed")
        prefix = self.model._voice_prefix
        with self.branch(kind):
            result = self.original_prompt(*args, **kwargs)
            if self.model._voice_prefix is not prefix:
                raise RuntimeError("Guidance changed speaker prefix")
        self.prompt_order.append(kind)
        if kind == "negative":
            # Different instruction lengths legitimately use different row positions.
            # Spoken history must still contain the same whole phrases in both lanes.
            if not self.in_step():
                raise RuntimeError("Paired branches retained different spoken histories")
            if self.model._continuing != self.shadow["_continuing"]:
                raise RuntimeError("Paired branches disagree on continuation")
        return result

    def make_cache(self):
        if not self.active:
            return self.original_cache()
        if self.prompt_order != ["conditional", "negative"] or len(self.cache_order) >= 2:
            raise RuntimeError("Native CFG cache order changed")
        kind = ("conditional", "negative")[len(self.cache_order)]
        with self.branch(kind):
            cache = self.original_cache()
        self.cache_order.append(kind)
        if kind == "negative":
            positive = self.model._speech_cache
            if cache is positive or len(cache) != len(positive) or any(a is b for a, b in zip(cache, positive)):
                raise RuntimeError("CFG branch caches alias")
            if any(a.keys is b.keys or a.values is b.values for a, b in zip(cache, positive)
                   if a.keys is not None and b.keys is not None):
                raise RuntimeError("CFG branch cache arrays alias")
        return cache

    def depth(self, first, hidden, *, unconditional_hidden, cfg_scale, **kwargs):
        if unconditional_hidden is None:
            return self.original_depth(first, hidden, unconditional_hidden=None, cfg_scale=cfg_scale, **kwargs)
        if not self.active or cfg_scale != self.scale:
            raise RuntimeError("Unexpected guided depth call")
        return self.paired_depth(first, hidden, unconditional_hidden=unconditional_hidden,
                                 cfg_scale=cfg_scale, **kwargs)

    def _finalize(self, words, complete):
        model = self.model
        states = [capture(model), self.shadow]
        safe = complete
        for state in states:
            cache = state["_speech_cache"]
            safe = safe and bool(model.context_rows and model.context_words
                                 and words <= model.context_words and cache
                                 and cache[0].size() <= model.context_rows
                                 and sum(c.keys.nbytes + c.values.nbytes for c in cache) <= model.context_bytes)
        if not safe:
            self.reset()
            return
        for state in states:
            state["_context_words"] += words
            if model.context_mode == "sliding":
                state["_speech_segments"].append((words, state["_speech_cache"][0].size() - state["_chunk_start_rows"]))
        activate(model, states[0])
        self.shadow = states[1]

    def generate(self, *args, **kwargs):
        if "cfg_scale" in kwargs:
            raise ValueError("Guidance owns cfg_scale")
        if self.active:
            raise RuntimeError("Nested guided generation")
        if self.scale == 1.0:
            # Exact old optimized path, including its RNG calls and compiled depth.
            yield from self.original_generate(*args, **kwargs)
            return
        if not kwargs.get("instruct"):
            raise ValueError("Guidance requires an explicit instruction")
        m = self.model
        if m.context_mode != "sliding":
            raise ValueError("Guidance requires the sliding speech context")
        words = len((args[0] if args else kwargs["text"]).split())
        m._next_words = words
        m._max_frames = kwargs.get("max_tokens", 750)
        self.prompt_order, self.cache_order = [], []
        self.active = True
        generator = self.native_generate(*args, cfg_scale=self.scale, **kwargs)
        complete, frames = False, 0
        try:
            for result in generator:
                frames += result.token_count
                yield result
            complete = 0 < frames < m._max_frames
        finally:
            try:
                generator.close()
            finally:
                self.active = False
                self._finalize(words, complete)


def build_paired_depth(model):
    """Frankie's cached depth decoding, run for both lanes and mixed with the native CFG formula."""
    import mlx.core as mx
    from mlx import nn
    from mlx_audio.lm.models.base import create_attention_mask
    from mlx_audio.lm.models.cache import KVCache
    from mlx_audio.lm.sample_utils import make_sampler

    def codes(first_codebook, conditional_hidden, unconditional_hidden, scale, temperature, top_p, top_k):
        decoder = model.depth_decoder.model
        cond_cache = [KVCache() for _ in decoder.layers]
        neg_cache = [KVCache() for _ in decoder.layers]
        for state in cond_cache + neg_cache:
            state.step = model.num_codebooks
        if decoder.backbone_hidden_state_projector is not None:
            conditional_hidden = decoder.backbone_hidden_state_projector(conditional_hidden)
            unconditional_hidden = decoder.backbone_hidden_state_projector(unconditional_hidden)
        valid = model.vocab_size
        k = min(top_k, valid) if top_k else 0
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=0 if k == valid else k)
        values = [first_codebook]
        for i, head in enumerate(model.depth_heads):
            token = (values[-1] + i * decoder.vocab_size).reshape(1, 1)
            embedded = decoder.embed_tokens(token)
            lane_logits = []
            for hidden, cache in ((conditional_hidden, cond_cache), (unconditional_hidden, neg_cache)):
                x = mx.concatenate([hidden[:, None], embedded], axis=1) if i == 0 else embedded
                x = decoder.inputs_embeds_projector(x)
                mask = create_attention_mask(x, cache[0])
                for layer, state in zip(decoder.layers, cache):
                    x = layer(x, mask, state)
                lane_logits.append(head(decoder.norm(x)[:, -1]))
            logits = lane_logits[1] + scale * (lane_logits[0] - lane_logits[1])
            logits = model._mask_reserved_codec_logits(logits)
            values.append(sampler(nn.log_softmax(logits[..., :valid], axis=-1)))
        return mx.concatenate(values)

    compiled = mx.compile(codes, inputs=mx.random.state, outputs=mx.random.state)

    def paired(first, hidden, *, unconditional_hidden, cfg_scale, temperature, top_p, top_k):
        return compiled(mx.array([first], dtype=mx.int32), hidden, unconditional_hidden,
                        cfg_scale, temperature, top_p, top_k).tolist()
    return paired
