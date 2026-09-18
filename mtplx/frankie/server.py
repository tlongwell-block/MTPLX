"""Run all Frankie models and the duplex web API in one MTPLX process."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path


def add_arguments(parser):
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18870)
    parser.add_argument("--mtp", type=int, default=3, choices=range(5))
    parser.add_argument("--http-slots", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--http-ctx-size", type=int, default=4096)
    parser.add_argument("--voice", type=Path)
    parser.add_argument("--voice-transcript")
    return parser


def serve(args):
    if not 128 <= args.http_ctx_size <= 131072:
        raise ValueError("--http-ctx-size must be between 128 and 131072.")
    token = os.environ.get("MTPLX_FRANKIE_TOKEN")
    if not token:
        raise ValueError("Set MTPLX_FRANKIE_TOKEN to a private access token.")
    from mtplx.profiles import apply_profile_env

    apply_profile_env("sustained")
    os.environ["MTPLX_COMPILED_VERIFY"] = "off"
    os.environ["MTPLX_COMPILE_AR_FORWARD"] = "0"
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse
    from .engine import Frankie
    from .session import Session

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frankie-models")
    owner = None
    engine = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal engine
        loop = asyncio.get_running_loop()

        def load():
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
            model = Frankie(args.brain, args.audio, mtp=args.mtp)
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
            return model

        engine = await loop.run_in_executor(executor, load)
        print(
            f"Frankie ready: one process, pid={os.getpid()}, MTP={args.mtp}, http://{args.host}:{args.port}",
            flush=True,
        )
        yield
        if owner is not None:
            await owner.close()
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(lifespan=lifespan)
    from .completions import attach_routes
    completion_service = attach_routes(app, lambda: engine, executor, token,
                                       slots=args.http_slots, context_tokens=args.http_ctx_size)

    @app.get("/health")
    async def health():
        return {
            "status": "ready" if engine else "loading",
            "model": "Frankie",
            "pid": os.getpid(),
            "mtp": args.mtp,
            "single_process": True,
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
