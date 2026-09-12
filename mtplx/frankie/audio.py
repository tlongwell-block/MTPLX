"""Existing MLX speech models and Frankie conditioning in the server process."""

from __future__ import annotations
import io
import json
import re
from pathlib import Path
from math import gcd

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .bridges import EarBridge, EarTone
from .vad import SileroVAD


def resample(pcm, source, target):
    if source == target:
        return np.asarray(pcm, dtype=np.float32)
    divisor = gcd(source, target)
    return resample_poly(pcm, target // divisor, source // divisor).astype(np.float32)


def _overlap_step(self, x):
    # Reference streaming-vocoder fix: carry the linear tail, not its bias,
    # otherwise every overlap sample receives the bias twice.
    y = self.conv(x)
    if self._overflow is not None:
        overlap = self._overflow.shape[1]
        y = mx.concatenate([y[:, :overlap] + self._overflow, y[:, overlap:]], axis=1)
    if self.trim_right > 0:
        self._overflow = y[:, -self.trim_right :] - (
            self.conv.bias if "bias" in self.conv else 0
        )
        y = y[:, : -self.trim_right]
    return y


class AudioModels:
    def __init__(self, directory):
        from parakeet_mlx.utils import from_config
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig
        from mlx_audio.tts.models.qwen3_tts import speech_tokenizer as codec

        root = Path(directory)
        cfg = json.loads((root / "frankie.json").read_text())
        self.weights = mx.load(str(root / cfg["conditioning"]))
        ear_path = root / cfg["ear"]
        self.ear = from_config(json.loads((ear_path / "config.json").read_text()))
        ew = mx.load(str(ear_path / "model.safetensors"))
        nn.quantize(
            self.ear,
            group_size=64,
            bits=8,
            class_predicate=lambda p, m: p + ".scales" in ew,
        )
        self.ear.load_weights(list(ew.items()))
        self.bridge = EarBridge(len(self.ear.vocabulary) + 1, 5120)
        self.tone = EarTone(self.weights["ear_tone.word_emb"])
        self.bridge.load_weights(self._weights("ear"))
        self.tone.load_weights(
            self._weights("ear_tone", exclude={"word_emb"}), strict=True
        )
        mouth_path = root / cfg["mouth"]
        self.qwen = Model(
            ModelConfig.from_dict(json.loads((mouth_path / "config.json").read_text()))
        )
        qw = mx.load(str(mouth_path / "model.safetensors"))
        nn.quantize(
            self.qwen,
            group_size=64,
            bits=8,
            class_predicate=lambda p, m: p + ".scales" in qw,
        )
        self.qwen.load_weights(list(qw.items()))
        mx.eval(self.qwen.parameters())
        self.qwen = Model.post_load_hook(self.qwen, mouth_path)
        self._speaker_encoder = self.qwen.extract_speaker_embedding
        self.offset = None
        self.qwen.extract_speaker_embedding = lambda *a, **k: self.speaker + (
            self.offset if self.offset is not None else 0
        )
        self._default_voice = {
            k: v for k, v in self.weights.items() if k.startswith("voice.")
        }
        self.set_voice(self._default_voice)
        codec.DecoderBlockUpsample.step = _overlap_step
        original = getattr(
            codec.Qwen3TTSSpeechTokenizerDecoder.streaming_step,
            "_frankie_original",
            codec.Qwen3TTSSpeechTokenizerDecoder.streaming_step,
        )

        def primed(decoder, codes):
            if decoder._transformer_cache is None:
                mx.eval(original(decoder, self.codes[:, :, -130:]))
            return original(decoder, codes)

        primed._frankie_original = original
        codec.Qwen3TTSSpeechTokenizerDecoder.streaming_step = primed
        mx.eval(
            self.ear.parameters(),
            self.bridge.parameters(),
            self.tone.parameters(),
            self.weights,
        )
        print(
            "Frankie ear/talker loaded in-process; 8-bit linear and embedding weights.",
            flush=True,
        )

    def _weights(self, prefix, exclude=frozenset()):
        return [
            (k[len(prefix) + 1 :], v)
            for k, v in self.weights.items()
            if k.startswith(prefix + ".") and k[len(prefix) + 1 :] not in exclude
        ]

    def make_vad(self):
        vad = SileroVAD()
        vad.load_weights(self._weights("vad"))
        return vad.prepare()

    def hear(self, pcm, rate=24000):
        from parakeet_mlx.audio import get_logmel

        audio = resample(pcm, rate, 16000)
        mel = get_logmel(mx.array(audio), self.ear.preprocessor_config)
        frames, _ = self.ear.encoder(mel[None] if mel.ndim == 2 else mel)
        rows = mx.concatenate([self.bridge(frames[0]), self.tone(frames[0])], axis=0)
        mx.eval(rows)
        return rows.astype(mx.bfloat16)

    def transcribe(self, pcm, rate=24000):
        from parakeet_mlx.audio import get_logmel

        mel = get_logmel(
            mx.array(resample(pcm, rate, 16000)), self.ear.preprocessor_config
        )
        return self.ear.generate(mel)[0].text

    def set_voice(self, values):
        self.voice = values
        self.codes = values["voice.codes"]
        self.speaker = values["voice.speaker"]
        self.reference_db = float(values["voice.rms_db"].item())
        self.reference = mx.zeros((24000,))
        self.reference_text = "packaged reference"
        self.qwen._icl_cache.clear()
        self.qwen._icl_cache[(self.reference_text, (24000, 0.0))] = (
            self.codes,
            values["voice.text_ids"],
        )

    def voice_from_wav(self, data, transcript=None):
        pcm, rate = sf.read(io.BytesIO(data), dtype="float32")
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=1)
        if not 1 <= len(pcm) / rate <= 30 or not np.isfinite(pcm).all():
            raise ValueError(
                "Voice reference must contain 1–30 seconds of finite audio."
            )
        pcm = resample(pcm, rate, 24000)
        if not transcript:
            transcript = self.transcribe(pcm)
        if not transcript.strip():
            raise ValueError("Voice reference needs audible speech or its transcript.")
        ref = mx.array(pcm)
        codes = self.qwen.speech_tokenizer.encode(ref[None, None])
        speaker = self._speaker_encoder(ref)
        ids = mx.array(
            self.qwen.tokenizer.encode(
                f"<|im_start|>assistant\n{transcript}<|im_end|>\n"
            )
        )[None, 3:-2]
        values = {
            "voice.codes": codes,
            "voice.speaker": speaker,
            "voice.text_ids": ids,
            "voice.rms_db": mx.array(20 * np.log10(np.sqrt(np.mean(pcm**2)) + 1e-9)),
        }
        mx.eval(values)
        self.set_voice(values)

    def expression(self, states):
        w = self.weights
        h = (mx.mean(states.astype(mx.float32), axis=0) - w["expression.mu"]) / w[
            "expression.sd"
        ]
        probabilities = mx.softmax(h @ w["expression.weight"].T + w["expression.bias"])
        gates = mx.clip((probabilities - 0.5) / 0.5, 0, 1)
        offset = gates @ w["expression.directions"].T
        mx.eval(offset)
        return offset.reshape(self.speaker.shape).astype(self.speaker.dtype)

    def speak(self, text, states, *, temperature=0.9):
        from num2words import num2words

        # Reuse Frankie's number expansion before the talker tokenizes words.
        text = re.sub(
            r"\b\d+\b", lambda m: num2words(int(m[0])) if len(m[0]) < 15 else m[0], text
        )
        self.offset = self.expression(states) if len(states) else None
        total_square = count = 0
        gain = 1.0
        generator = self.qwen.generate(
            text,
            ref_audio=self.reference,
            ref_text=self.reference_text,
            stream=True,
            streaming_interval=0.08,
            temperature=temperature,
            max_tokens=max(75, 12 * len(text.split())),
            verbose=False,
        )
        try:
            for result in generator:
                pcm = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if not np.isfinite(pcm).all():
                    raise RuntimeError("Speech decoder produced non-finite audio.")
                total_square += float(np.square(pcm).sum())
                count += len(pcm)
                rms = np.sqrt(total_square / max(1, count))
                gain = min(
                    gain,
                    0.9 / max(float(np.abs(pcm).max(initial=0)), 1e-9),
                    10 ** ((self.reference_db + 3) / 20) / max(rms, 1e-9),
                )
                yield pcm * gain
        finally:
            generator.close()
            self.offset = None
