"""Run the Frankie realtime demo the way it runs live.

    MTPLX_FRANKIE_TOKEN=... python -m mtplx.frankie.demo \\
        --brain BRAIN --audio AUDIO --voice VOICE.wav \\
        --delivery ASSETS/delivery --maai ASSETS/maai \\
        --watermark ASSETS/generator_streaming.pth [--log RUN]

On top of the plain server this installs, in order: the AudioSeal watermark
on final speech, MaAI turn-taking, one fixed sampling seed per session,
first-audio latency telemetry, and brain-led delivery with the held voice.
Server options and MTPLX_* settings default to the live demo's; anything
given on the command line or already in the environment wins.
"""
import argparse
import functools
import json
import os
from pathlib import Path

# The live demo's settings. Set before numpy or torch load, since some are read once at import.
SETTINGS = dict(
    MTPLX_FRANKIE_CODEC_CONTEXT="convolution",
    MTPLX_FRANKIE_SPEECH_CONTEXT_MODE="sliding",
    MTPLX_FRANKIE_SPEECH_CONTEXT_WORDS="250",
    MTPLX_FRANKIE_SPEECH_CONTEXT_BYTES="384000000",
    MTPLX_FRANKIE_SPEECH_HOLD_WORDS="40",
    MTPLX_FRANKIE_BREEZE_CONTEXT_ROWS="4096",
    MTPLX_FRANKIE_IDLE_HISTORY_PREFILL="1",
    MTPLX_FRANKIE_PROMPT_SEGMENTS="1",
    MTPLX_FRANKIE_FIRST_PHRASE_LOOKAHEAD="0",
    MTPLX_FRANKIE_BUFFERED_PHRASE_LOOKAHEAD="0",
    MTPLX_FRANKIE_CPU_VAD_PREP="0",
    MTPLX_FRANKIE_GPU_RESIDENCY="1",
    MTPLX_GPU_KEEPALIVE="1",
    MTPLX_SESSION_BANK_MAX_BYTES="16G",
    MTPLX_SESSION_BANK_PER_SESSION_BYTES="16G",
    NO_TORCH_COMPILE="1",
    OMP_NUM_THREADS="2",
    TOKENIZERS_PARALLELISM="false",
)
SEED = 1900


def seed_sessions(engine_class, seed=SEED):
    """Seed MLX once at each session's first reply, so a session's first sampling is repeatable."""
    original, seeded = engine_class.respond, set()

    @functools.wraps(original)
    def respond(self, items, settings, emit, abort, *, session_id):
        if session_id not in seeded:
            import mlx.core as mx
            mx.random.seed(seed)
            seeded.add(session_id)
        return original(self, items, settings, emit, abort, session_id=session_id)

    engine_class.respond = respond


def main(argv=None):
    for name, value in SETTINGS.items():
        os.environ.setdefault(name, value)
    from mtplx.frankie.server import add_arguments, configure_runtime, serve

    parser = add_arguments(argparse.ArgumentParser(description=__doc__,
                                                   formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.set_defaults(brain_interface="text", profile="turbo", mtp=2, http_slots=1, http_ctx_size=65536)
    parser.add_argument("--delivery", type=Path, required=True,
                        help="Brain-led delivery assets: directions, calibration, adapter, rows/.")
    parser.add_argument("--maai", type=Path, required=True,
                        help="MaAI BC-Det weights: bc_det.pt, mimi.onnx, mimi.json.")
    parser.add_argument("--watermark", type=Path, required=True,
                        help="AudioSeal generator_streaming.pth (16 bits).")
    parser.add_argument("--log", type=Path, help="Folder for delivery.jsonl, phrase states and watermark.jsonl.")
    args = parser.parse_args(argv)

    # One CPU thread for torch (AudioSeal and MaAI), set before either runs anything.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configure_runtime(args.brain, args.profile, brain_interface=args.brain_interface)
    from mtplx.frankie import engine
    from . import delivery, maai, telemetry, watermark

    if args.log:
        args.log.mkdir(parents=True, exist_ok=True)

    def report(row):
        if args.log:
            with open(args.log / "watermark.jsonl", "a") as f:
                f.write(json.dumps({**row, "watermark": "residual16", "pid": os.getpid()}) + "\n")

    watermark.install(engine.Frankie, watermark.audioseal_factory(args.watermark), report)
    maai.install(maai.Detector(args.maai))
    seed_sessions(engine.Frankie)
    telemetry.install()
    delivery.install(engine.Frankie, args.delivery, log=args.log)
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
