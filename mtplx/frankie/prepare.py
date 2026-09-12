"""Prepare native MLX audio weights and conditioning from a Frankie package.

This never writes a standalone reference recording. The reference is decoded in
memory to obtain the existing speech model's speaker embedding and codec context.
"""

import argparse
import io
import json
from pathlib import Path
import shutil
import tempfile


def conditioning(gguf_path, mouth):
    """Read the learned adapters and voice conditioning from the same package."""
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from gguf import GGUFReader

    gg = GGUFReader(str(gguf_path))
    tensors = {t.name: t for t in gg.tensors}
    data = {}
    for name, tensor in tensors.items():
        if name.startswith("frankie.bridge."):
            data[name.removeprefix("frankie.bridge.")] = mx.array(tensor.data.copy())
        elif name.startswith("frankie.vad."):
            values = tensor.data
            if values.ndim == 3:
                values = values.transpose(0, 2, 1)  # GGML conv -> MLX conv
            data[name.removeprefix("frankie.vad.")] = mx.array(values.copy())
    with tempfile.NamedTemporaryFile(suffix=".gguf") as temporary:
        temporary.write(tensors["assets.expression.gguf"].data.tobytes())
        temporary.flush()
        expression = GGUFReader(temporary.name)
        data.update({t.name: mx.array(t.data.copy()) for t in expression.tensors})
    pcm, sr = sf.read(
        io.BytesIO(tensors["assets.voice.wav"].data.tobytes()), dtype="float32"
    )
    if sr != 24000 or pcm.ndim != 1:
        raise ValueError("Packaged voice must be mono 24 kHz audio.")
    field = gg.fields["frankie.voice.ref_text"]
    ref_text = bytes(field.parts[field.data[-1]]).decode("utf-8")
    data.update(
        {
            "voice.codes": mx.array(tensors["assets.voice.codes"].data.copy()).T[None],
            "voice.text_ids": mx.array(
                mouth.tokenizer.encode(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n")
            )[None, 3:-2],
            "voice.speaker": mouth.extract_speaker_embedding(mx.array(pcm)),
            "voice.rms_db": mx.array(20 * np.log10(np.sqrt(np.mean(pcm**2)) + 1e-9)),
        }
    )
    return data


def quantize(model):
    import mlx.nn as nn

    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda p, m: not p.startswith("speech_tokenizer.")
        and isinstance(m, (nn.Linear, nn.Embedding))
        and m.weight.shape[-1] % 64 == 0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "ear", "mouth", "output"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = ap.parse_args()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from parakeet_mlx import from_pretrained
    from mlx_audio.tts.utils import load_model

    a.output.mkdir(parents=True, exist_ok=False)
    for name, source, loader in [
        ("ear", a.ear, from_pretrained),
        ("mouth", a.mouth, load_model),
    ]:
        dest = a.output / name
        dest.mkdir()
        for p in source.iterdir():
            if p.is_file() and p.suffix in {".json", ".txt", ".model", ".jinja"}:
                shutil.copyfile(p, dest / p.name)
        model = loader(str(source))
        codec = model.pop("speech_tokenizer", None) if name == "mouth" else None
        quantize(model)
        weights = {
            k: v
            for k, v in tree_flatten(model.parameters())
            if not k.startswith("speech_tokenizer.")
        }
        mx.save_safetensors(str(dest / "model.safetensors"), weights)
        cfg = json.loads((dest / "config.json").read_text())
        cfg["quantization"] = {"group_size": 64, "bits": 8}
        (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        if name == "mouth":
            model.speech_tokenizer = codec
            shutil.copytree(source / "speech_tokenizer", dest / "speech_tokenizer")
            mouth = model
        del model, weights
        print("Prepared", name, "8-bit weights", flush=True)
    data = conditioning(a.gguf, mouth)
    mx.save_safetensors(str(a.output / "conditioning.safetensors"), data)
    (a.output / "frankie.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "ear": "ear",
                "mouth": "mouth",
                "conditioning": "conditioning.safetensors",
                "feature_layer": 16,
                "audio_quantization": {
                    "linear_and_embedding_bits": 8,
                    "group_size": 64,
                    "codec": "source precision",
                },
                "voice": {"conditioning_only": True},
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Prepared Frankie audio package; reference waveform was not written.",
        flush=True,
    )


if __name__ == "__main__":
    main()
