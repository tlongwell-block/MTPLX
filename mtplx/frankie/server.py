"""Run all Frankie models and the duplex web API in one MTPLX process."""

import argparse
import asyncio
import hmac
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace


def add_arguments(parser):
    from mtplx.profiles import DEFAULT_PROFILE_NAME, PROFILE_CHOICES
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18870)
    parser.add_argument("--mtp", type=int, default=3, choices=range(5))
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--brain-interface", choices=("neural", "text"), default="neural")
    parser.add_argument("--http-slots", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--http-ctx-size", type=int, default=4096)
    parser.add_argument("--voice", type=Path)
    parser.add_argument("--voice-transcript")
    parser.add_argument("--mlx-cache-limit", help="Freed-buffer cache limit, e.g. 8G; off uses MLX defaults.")
    return parser


def configure_runtime(brain, profile, *, brain_interface="neural"):
    from mtplx.profiles import apply_profile_env
    overrides = None
    if brain_interface == "text":
        from types import SimpleNamespace
        from mtplx.commands.public import _in_process_runtime_env_overrides
        overrides = _in_process_runtime_env_overrides(
            SimpleNamespace(verify_strategy="capture_commit"), str(brain), generation_mode="mtp"
        )
    apply_profile_env(profile, runtime_env_overrides=overrides)
    if brain_interface == "neural":
        os.environ["MTPLX_COMPILED_VERIFY"] = "off"
        os.environ["MTPLX_COMPILE_AR_FORWARD"] = "0"


def serve(args):
    if not 128 <= args.http_ctx_size <= 131072:
        raise ValueError("--http-ctx-size must be between 128 and 131072.")
    token = os.environ.get("MTPLX_FRANKIE_TOKEN")
    if not token:
        raise ValueError("Set MTPLX_FRANKIE_TOKEN to a private access token.")
    configure_runtime(args.brain, args.profile, brain_interface=args.brain_interface)
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse

    from .engine import Frankie
    from .session import Session
    from mtplx.server.openai import _gpu_keepalive_enabled

    # Opt-in until live parity and paced voice testing qualify residency.
    residency_enabled = _gpu_keepalive_enabled() and os.environ.get("MTPLX_FRANKIE_GPU_RESIDENCY", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if residency_enabled:
        from mtplx.model_scheduler import ModelWorkScheduler
        executor = ModelWorkScheduler(name="frankie-models")
    else:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frankie-models")
    residency = SimpleNamespace(model_scheduler=executor, metal_memory_caps={},
                                gpu_keepalive={}, previous_limits={}, failure=None)
    residency_started = False
    owner = None
    engine = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal engine
        loop = asyncio.get_running_loop()

        def load():
            import mlx.core as mx

            from mtplx.server.openai import _configure_mlx_cache_limit

            mx.set_default_device(mx.gpu)
            _configure_mlx_cache_limit(args)
            model = Frankie(args.brain, args.audio, mtp=args.mtp, brain_interface=args.brain_interface)
            if args.voice:
                model.audio.voice_from_wav(
                    args.voice.read_bytes(), args.voice_transcript
                )
                model.audio._default_voice = model.audio.voice
            from threading import Event

            model.respond(
                [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Say hello briefly."}
                        ],
                    }
                ],
                {
                    "instructions": "You are Frankie. Answer briefly.",
                    "max_output_tokens": 32,
                    "output_modalities": ["audio"],
                    "thinking": "off",
                    "temperature": 0,
                },
                lambda *a: None,
                Event(),
                session_id="warmup",
            )
            model.bank.clear()
            if residency_enabled:
                from mtplx.server.openai import (
                    _apply_metal_memory_caps, _detect_total_ram_bytes_for_metal_caps,
                    _resident_floor_margin_bytes,
                )
                ram, _ = _detect_total_ram_bytes_for_metal_caps()
                residency.metal_memory_caps = _apply_metal_memory_caps(
                    total_ram_bytes=ram,
                    minimum_resident_bytes=(mx.get_active_memory()
                                            + _resident_floor_margin_bytes(ram)),
                    previous_limits=residency.previous_limits,
                )
            return model

        try:
            engine = await loop.run_in_executor(executor, load)
            print(
                f"Frankie ready: one process, pid={os.getpid()}, MTP={args.mtp}, http://{args.host}:{args.port}",
                flush=True,
            )
            yield
        finally:
            try:
                if owner is not None:
                    await owner.close()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

    def residency_fallback(receipt):
        """Restore this feature's wiring on the model owner, preserving old caps."""
        from mtplx.server.openai import _set_metal_memory_limit
        executor.disarm_idle_keepalive()
        receipt["enabled"] = False
        caps = residency.metal_memory_caps
        if caps.get("wired_limit_api"):
            try:
                import mlx.core as mx
                previous = residency.previous_limits["set_wired_limit"]
                _set_metal_memory_limit(mx, "set_wired_limit", previous)
                caps["wired_limit_bytes"] = previous
                receipt["restored_wired_limit_bytes"] = previous
            except Exception as exc:
                residency.failure = f"residency_restore_failed:{type(exc).__name__}"
                receipt["reason"] = residency.failure
                executor.shutdown(wait=False, cancel_futures=True)
        residency.gpu_keepalive = receipt

    def residency_failed():
        residency_fallback({"reason": "keepalive_runtime_failure"})

    def get_engine():
        nonlocal residency_started
        # Called by get_service only after authenticated HTTP/voice admission.
        # Arming and its tiny MLX allocation run on the same model owner.
        if residency_enabled and not residency_started:
            residency_started = True
            def arm():
                from mtplx.server.openai import _arm_gpu_keepalive
                try:
                    receipt = _arm_gpu_keepalive(residency, on_failure=residency_failed)
                except Exception as exc:
                    receipt = {"enabled": False, "reason": f"arm_failed:{type(exc).__name__}"}
                if receipt["enabled"]:
                    residency.gpu_keepalive = receipt
                else:
                    residency_fallback(receipt)
            executor.submit(arm)
        return engine

    app = FastAPI(lifespan=lifespan)
    from .completions import attach_routes
    completion_service = attach_routes(app, get_engine, executor, token,
                                       slots=args.http_slots, context_tokens=args.http_ctx_size)

    @app.get("/health")
    async def health():
        from mtplx.server.openai import _gpu_keepalive_health
        return {
            "status": "failed" if residency.failure else "ready" if engine else "loading",
            "model": "Frankie",
            "pid": os.getpid(),
            "mtp": args.mtp,
            "brain_interface": args.brain_interface,
            "profile": args.profile,
            "single_process": True,
            "gpu_keepalive": _gpu_keepalive_health(residency),
            "metal_memory_caps": residency.metal_memory_caps,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return Path(__file__).with_name("page.html").read_text()

    @app.get("/audio-worklet.mjs")
    async def worklet():
        return FileResponse(
            Path(__file__).with_name("audio-worklet.mjs"), media_type="text/javascript"
        )

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket):
        nonlocal owner
        auth = ws.headers.get("authorization", "").removeprefix("Bearer ")
        protocols = ws.headers.get("sec-websocket-protocol", "").split(",")
        for value in protocols:
            value = value.strip()
            if value.startswith("openai-insecure-api-key."):
                auth = value.split(".", 1)[1]
        if not hmac.compare_digest(auth, token):
            await ws.close(code=1008, reason="Invalid access token.")
            return
        if owner is not None:
            await ws.close(code=1013, reason="Another conversation owns Frankie.")
            return
        session = Session(engine, executor, ws)
        from .perception import RealtimeListener
        session.listener = RealtimeListener(session, completion_service())
        owner = session
        sender = None
        try:
            await ws.accept(
                subprotocol="realtime"
                if any(p.strip() == "realtime" for p in protocols)
                else None
            )
            sender = asyncio.create_task(session.send_events())
            session.event("session.created", session=session.info())
            while True:
                try:
                    await session.handle(await ws.receive_json())
                except (ValueError, KeyError, TypeError) as exc:
                    session.event(
                        "error",
                        error={"type": "invalid_request_error", "message": str(exc)},
                    )
                except RuntimeError as exc:
                    session.event(
                        "error", error={"type": "server_error", "message": str(exc)},
                    )
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()
            if sender is not None:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            if owner is session:
                owner = None

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        ws_max_size=18 * 1024**2,
        log_level="info",
    )
    return 0


def main():
    return serve(
        add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    )


if __name__ == "__main__":
    main()
