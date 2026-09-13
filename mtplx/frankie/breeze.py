"""Frankie text-to-speech using mlx-audio's Breeze model and cached depth decoding."""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.sample_utils import make_sampler
from mlx_audio.tts.models.breeze_tts import Model, ModelConfig


class TextEmbedding(nn.Module):
    """Keep Breeze's scaling and EOI override with a quantizable lookup."""

    def __init__(self, original):
        super().__init__()
        self.embedding = nn.Embedding(*original.weight.shape)
        self.embedding.weight = original.weight
        self.eoi_embedding = original.eoi_embedding
        self.eoi_token_index = original.eoi_token_index
        self.scale = original.weight.shape[-1] ** 0.5

    def __call__(self, ids):
        return mx.where(
            (ids == self.eoi_token_index)[..., None],
            self.eoi_embedding,
            self.embedding(ids) * self.scale,
        )


class BreezeModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.text_encoder.embed_tokens = TextEmbedding(self.text_encoder.embed_tokens)
        head = self.depth_decoder.pop("codebooks_head").weight
        self.depth_heads = [
            nn.Linear(head.shape[1], head.shape[2], bias=False)
            for _ in range(head.shape[0])
        ]
        self._voice_prefix = None
        self.context_rows = int(
            os.environ.get("MTPLX_FRANKIE_BREEZE_CONTEXT_ROWS", "2048")
        )
        self.context_words = int(
            os.environ.get("MTPLX_FRANKIE_SPEECH_CONTEXT_WORDS", "100")
        )
        if not 0 <= self.context_words <= 1000:
            raise ValueError("Breeze speech context must be zero to 1000 words.")
        if self.context_rows and not 1024 <= self.context_rows <= 8192:
            raise ValueError("Breeze speech context must be zero or 1024-8192 rows.")
        self._new_cache = self.backbone_model.make_cache
        self.backbone_model.make_cache = self._generation_cache
        self.reset_speech_context()

    def reset_speech_context(self):
        self._speech_cache = None
        self._context_words = 0
        self._continuing = False
        self._max_frames = 0

    def _generation_cache(self):
        if not self._continuing:
            self._speech_cache = self._new_cache()
            self._context_words = 0
        return self._speech_cache

    def generate(self, *args, **kwargs):
        words = len((args[0] if args else kwargs["text"]).split())
        if self._context_words + words > self.context_words:
            self.reset_speech_context()
        self._max_frames = kwargs.get("max_tokens", 750)
        complete, frames = False, 0
        try:
            for result in super().generate(*args, **kwargs):
                frames += result.token_count
                yield result
            complete = 0 < frames < self._max_frames
        finally:
            if (
                not complete
                or not self.context_rows
                or not self.context_words
                or words > self.context_words
                or self._speech_cache is None
                or self._speech_cache[0].offset > self.context_rows
                or sum(s.keys.nbytes + s.values.nbytes for s in self._speech_cache)
                > 100_000_000
            ):
                self.reset_speech_context()
            else:
                self._context_words += words

    @staticmethod
    def sanitize(weights):
        values = Model.sanitize(weights)
        name = "text_encoder.embed_tokens.weight"
        if name in values:
            values["text_encoder.embed_tokens.embedding.weight"] = values.pop(name)
        name = "depth_decoder.codebooks_head.weight"
        if name in values:
            for i, weight in enumerate(values.pop(name)):
                values[f"depth_heads.{i}.weight"] = weight.T
        return values

    def _prompt_embeddings(self, *args, **kwargs):
        target = super()._prompt_embeddings(*args, **kwargs)
        if self._voice_prefix is None:
            raise ValueError("Breeze voice conditioning was not initialized.")
        cached = 0 if self._speech_cache is None else self._speech_cache[0].offset
        self._continuing = bool(
            self.context_rows
            and self.context_words
            and cached
            and cached + target.shape[1] + 1 + self._max_frames <= self.context_rows
        )
        if self._continuing:
            eos = mx.full((1, 1, self.num_codebooks), self.config.codebook_eos_token_id)
            prefix = self.backbone_model.embed_tokens(eos)
        else:
            prefix = self._voice_prefix[None].astype(target.dtype)
        if self.context_rows:
            print(
                f"Breeze context: cached={cached if self._continuing else 0} new={prefix.shape[1] + target.shape[1]}",
                flush=True,
            )
        return mx.concatenate([prefix, target], axis=1)

    def reference_prefix(self, pcm, transcript):
        ids = self._text_ids(f"[S0]{transcript}")
        text = self.text_encoder_proj(self.text_encoder(ids[None]))
        codes = self._encode_reference(mx.array(pcm))
        eos = mx.full((1, 1, self.num_codebooks), self.config.codebook_eos_token_id)
        return mx.concatenate(
            [
                text,
                self.backbone_model.embed_tokens(codes),
                self.backbone_model.embed_tokens(eos),
            ],
            axis=1,
        )[0]

    def _depth_tokens(
        self,
        first_codebook,
        conditional_hidden,
        *,
        unconditional_hidden,
        cfg_scale,
        temperature,
        top_p,
        top_k,
    ):
        if unconditional_hidden is not None:
            raise ValueError("Frankie uses one unguided Breeze branch.")
        model = self.depth_decoder.model
        cache = [KVCache() for _ in model.layers]
        hidden = conditional_hidden
        if model.backbone_hidden_state_projector is not None:
            hidden = model.backbone_hidden_state_projector(hidden)
        # Match upstream's codec-only distribution, but keep depth samples on
        # the GPU until the frame is complete instead of synchronizing each head.
        valid = self.vocab_size
        effective_top_k = min(top_k, valid) if top_k else 0
        if effective_top_k == valid:
            effective_top_k = 0
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=effective_top_k)
        codes = [mx.array([first_codebook], dtype=mx.int32)]
        for i, head in enumerate(self.depth_heads):
            token = (codes[-1] + i * model.vocab_size).reshape(1, 1)
            x = model.embed_tokens(token)
            if i == 0:
                x = mx.concatenate([hidden[:, None], x], axis=1)
            x = model.inputs_embeds_projector(x)
            mask = create_attention_mask(x, cache[0])
            for layer, state in zip(model.layers, cache):
                x = layer(x, mask, state)
            logits = self._mask_reserved_codec_logits(head(model.norm(x)[:, -1]))
            codes.append(sampler(nn.log_softmax(logits[..., :valid], axis=-1)))
        return mx.concatenate(codes).tolist()


def load_breeze(directory):
    root = Path(directory)
    cfg = json.loads((root / "config.json").read_text())
    model = BreezeModel(ModelConfig.from_dict(cfg))
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        weights.update(mx.load(str(path)))
    weights = model.sanitize(weights)
    if "quantization" in cfg:
        nn.quantize(
            model,
            **cfg["quantization"],
            class_predicate=lambda p, m: p + ".scales" in weights,
        )
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model = Model.post_load_hook(model, root)
    model.eval()
    return model


class BreezeMouth:
    def __init__(self, directory, weights):
        self.model = load_breeze(directory)
        self.weights = weights
        self._default_voice = {
            "voice.prefix": weights["breeze.voice_prefix"],
            "voice.rms_db": weights["voice.rms_db"],
        }
        self.set_voice(self._default_voice)

    def set_voice(self, values):
        self.model.reset_speech_context()
        self.voice = values
        self.model._voice_prefix = values["voice.prefix"]
        self.reference_db = float(values["voice.rms_db"].item())

    def voice_from_pcm(self, pcm, transcript):
        model = self.model
        prefix = model.reference_prefix(pcm, transcript)
        values = {
            "voice.prefix": prefix,
            "voice.rms_db": mx.array(20 * np.log10(np.sqrt(np.mean(pcm**2)) + 1e-9)),
        }
        mx.eval(values)
        self.set_voice(values)

    def instruction(self, states):
        instruction = "Speak clearly and naturally"
        if not len(states):
            return instruction + "."
        if states.ndim != 2 or states.shape[1] != 5120 or len(states) > 512:
            raise ValueError("Invalid emotion-head states.")
        if not bool(mx.all(mx.isfinite(states)).item()):
            raise ValueError("Non-finite emotion-head states.")
        w = self.weights
        h = (states.astype(mx.float32).mean(axis=0) - w["expression.mu"]) / w[
            "expression.sd"
        ]
        probabilities = mx.softmax(h @ w["expression.weight"].T + w["expression.bias"])
        winner = int(mx.argmax(probabilities).item())
        if winner < 3 and float(probabilities[winner]) > 0.5:
            instruction += " with a " + ("angry", "happy", "sad")[winner] + " tone"
        return instruction + "."

    def speak(self, text, states, *, temperature=0.9):
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 8192:
            raise ValueError("Breeze requires bounded nonempty spoken text.")
        model = self.model
        token_count = len(model._text_ids(text))
        if token_count > 512:
            raise ValueError("Breeze spoken span exceeds 512 tokens.")
        generator = model.generate(
            text,
            instruct=self.instruction(states),
            stream=True,
            streaming_interval=0.08,
            temperature=temperature,
            max_tokens=min(384, max(75, token_count * 12)),
        )
        gain, energy, samples = 1.0, 0.0, 0
        try:
            for result in generator:
                pcm = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if not np.isfinite(pcm).all():
                    raise RuntimeError("Breeze decoder produced non-finite audio.")
                energy += float(np.square(pcm).sum())
                samples += len(pcm)
                rms = np.sqrt(energy / max(1, samples))
                gain = min(
                    gain,
                    0.9 / max(float(np.abs(pcm).max(initial=0)), 1e-9),
                    10 ** ((self.reference_db + 3) / 20) / max(rms, 1e-9),
                )
                yield pcm * gain
        finally:
            generator.close()
            model.audio_tokenizer.decoder.reset_streaming_state()
