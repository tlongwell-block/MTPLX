"""Exercise a running single-process Frankie server, including live PCM ingress."""

import argparse
import asyncio
import json
import base64
import time
import wave
import re
import os
import io
from pathlib import Path
import websockets
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--url", default="ws://127.0.0.1:18870/v1/realtime")
ap.add_argument(
    "--audio",
    type=Path,
    required=True,
    help="Mono PCM16 24 kHz WAV asking what two plus two is",
)
ap.add_argument("--output", type=Path, default=Path("frankie-results.json"))
a = ap.parse_args()
token = os.environ["MTPLX_FRANKIE_TOKEN"]
with wave.open(str(a.audio)) as w:
    assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
    pcm = w.readframes(w.getnframes())

fixture = Image.new("RGB", (420, 180), "white")
draw = ImageDraw.Draw(fixture)
for x, color in [(20, "red"), (160, "green"), (300, "blue")]:
    draw.rectangle((x, 30, x + 100, 130), fill=color)
png = io.BytesIO()
fixture.save(png, format="PNG")
results = []


async def main():
    async with websockets.connect(
        a.url,
        additional_headers={"Authorization": "Bearer " + token},
        max_size=16 * 1024**2,
    ) as ws:

        async def send(**x):
            await ws.send(json.dumps(x))

        async def until(kind):
            while True:
                e = json.loads(await asyncio.wait_for(ws.recv(), 90))
                if e["type"] == "error":
                    raise AssertionError(e)
                if e["type"] == kind:
                    return e

        async def update(**s):
            await send(type="session.update", session=s)
            await until("session.updated")

        async def add(text=None, image=None, audio=None):
            c = []
            if text:
                c.append({"type": "input_text", "text": text})
            if image:
                c.append(
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(image).decode(),
                    }
                )
            if audio:
                c.append(
                    {"type": "input_audio", "audio": base64.b64encode(audio).decode()}
                )
            await send(
                type="conversation.item.create",
                item={"type": "message", "role": "user", "content": c},
            )
            await until("conversation.item.created")

        async def response(name):
            t = time.monotonic()
            await send(type="response.create")
            frames = []
            events = []
            metrics = None
            first = None
            while True:
                e = json.loads(await asyncio.wait_for(ws.recv(), 90))
                events.append(e["type"])
                if e["type"] == "error":
                    raise AssertionError(e)
                if e["type"] == "response.output_audio.delta":
                    frames.append(base64.b64decode(e["delta"]))
                    first = first or time.monotonic() - t
                if e["type"] == "frankie.metrics":
                    metrics = e["metrics"]
                if e["type"] == "response.done":
                    r = e["response"]
                    break
            row = {
                "name": name,
                "seconds": time.monotonic() - t,
                "first_audio": first,
                "audio_seconds": sum(map(len, frames)) / 48000,
                "response": r,
                "metrics": metrics,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            assert r["status"] == "completed", r
            return r

        def text(r):
            return " ".join(
                p.get("text", p.get("transcript", ""))
                for i in r["output"]
                if i["type"] == "message"
                for p in i["content"]
            )

        await until("session.created")
        await update(
            temperature=0,
            max_output_tokens=128,
            audio={"input": {"turn_detection": None}},
            output_modalities=["text"],
        )
        await add("My checkpoint is crimson-orbit-7429. Remember it and say okay.")
        await response("text")
        await add("What is my checkpoint?")
        r = await response("memory")
        assert "7429" in text(r), r
        await add("What is in this image?", image=png.getvalue())
        r = await response("image")
        assert any(k in text(r).lower() for k in ["red", "blue"]), r
        await update(output_modalities=["audio"])
        await add("What is seven times eight? Answer briefly.")
        r = await response("speech")
        assert "56" in text(r) or "fifty" in text(r).lower(), r
        await add(audio=pcm)
        r = await response("audio-input")
        assert re.search(r"\b(four|4)\b", text(r).lower()), r
        await update(
            output_modalities=["text"], thinking="minimal", max_output_tokens=256
        )
        await add("What is my checkpoint?")
        r = await response("thinking")
        assert "7429" in text(r), r
        await update(
            thinking="off",
            tools=[
                {
                    "type": "function",
                    "name": "lookup_checkpoint",
                    "description": "Look up a checkpoint.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                }
            ],
        )
        await add("Call lookup_checkpoint with key lighthouse.")
        r = await response("tool-call")
        calls = [i for i in r["output"] if i["type"] == "function_call"]
        assert len(calls) == 1, r
        await send(
            type="conversation.item.create",
            item={
                "type": "function_call_output",
                "call_id": calls[0]["call_id"],
                "output": '{"code":"cedar-58"}',
            },
        )
        await until("conversation.item.created")
        r = await response("tool-result")
        assert "58" in text(r), r
        # Keep sending real-time microphone frames while speech is being produced,
        # then interrupt and submit the captured utterance as the next turn.
        await update(
            thinking="off", tools=[], output_modalities=["audio"], max_output_tokens=512
        )
        await add("Describe a peaceful garden in ten sentences.")
        await send(type="response.create")
        first = await until("response.output_audio.delta")
        started = time.monotonic()
        for offset in range(0, len(pcm), 960):
            await send(
                type="input_audio_buffer.append",
                audio=base64.b64encode(pcm[offset : offset + 960]).decode(),
            )
            if offset == 0:
                await send(type="response.cancel", response_id=first["response_id"])
                await send(
                    type="conversation.item.truncate",
                    item_id=first["item_id"],
                    audio_end_ms=0,
                )
            await asyncio.sleep(
                max(0, started + (offset + 960) / 48000 - time.monotonic())
            )
        cancelled = await until("response.done")
        assert cancelled["response"]["status"] == "cancelled"
        await send(type="conversation.item.retrieve", item_id=first["item_id"])
        heard = await until("conversation.item.retrieved")
        assert (
            heard["item"]["content"][0]["transcript"].strip()
            == "[interrupted by the user]"
        )
        await send(type="input_audio_buffer.commit")
        await until("input_audio_buffer.committed")
        r = await response("duplex-cancel-and-continue")
        assert re.search(r"\b(four|4)\b", text(r).lower()), r
        # A user-supplied reference is encoded in this same server process.
        await send(
            type="frankie.voice.update",
            wav=base64.b64encode(a.audio.read_bytes()).decode(),
        )
        await until("frankie.voice.updated")
        await add("Say hello in one short sentence.")
        await response("custom-voice")
    a.output.write_text(json.dumps(results, indent=2))
    print("PASS: " + str(a.output))


asyncio.run(main())
