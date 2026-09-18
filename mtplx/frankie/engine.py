"""One inference owner for brain, vision, learned audio input and speech output."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections import deque
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path

import mlx.core as mx

from mtplx.features import CommittedFeatures
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank
from mtplx.vision import load_vision_tower, vision_spec_for_model_dir
from mtplx.vision.processing import decode_image, preprocess_images
from mtplx.vision.splice import VisionSplice

from .audio import AudioModels
from .sampling import brain_sampler, thinking_guard


class Frankie:
    def __init__(self, brain, audio, *, mtp=3):
        self.runtime = load(brain, mtp=True)
        self.tokenizer = self.runtime.tokenizer
        self.audio = AudioModels(audio)
        self.vision = load_vision_tower(brain)
        self.vision_spec = vision_spec_for_model_dir(brain)
        self.image_config = json.loads(
            (Path(brain) / "preprocessor_config.json").read_text()
        )
        self.mtp = mtp
        self.bank = SessionBank(
            max_entries=6, max_bytes=8 * 1024**3, per_session_max_bytes=8 * 1024**3
        )
        print("Powered by MTPLX — https://github.com/youssofal/MTPLX", flush=True)

    def image(self, data):
        pixels, grid = preprocess_images([decode_image(data)], self.image_config)
        rows, deep = self.vision(pixels, grid)
        if deep:
            raise ValueError(
                "This Frankie checkpoint requires a vision tower without deepstack."
            )
        mx.eval(rows)
        return rows

    def release(self, session_id):
        self.bank.clear(session_id=session_id)
        self.audio.set_voice(self.audio._default_voice)

    def prepare_audio(self, item):
        transcripts = []
        for index, part in enumerate(item.get("content", [])):
            if part["type"] == "input_audio":
                if "_rows" not in part:
                    part["_rows"], part["_transcript"] = self.audio.hear(
                        part["_pcm"], part.get("_rate", 24000)
                    )
                transcripts.append((item, index, part["_transcript"]))
        return transcripts

    def prompt(self, items, settings, *, generation_prompt=True, emit=None):
        messages = [{"role": "system", "content": settings["instructions"]}]
        media = []
        for item in items:
            for transcript in self.prepare_audio(item):
                if emit is not None:
                    emit("input_transcript", transcript)
            if item["type"] == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": item["output"],
                    }
                )
                continue
            if item["type"] == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item["call_id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.loads(item["arguments"]),
                                },
                            }
                        ],
                    }
                )
                continue
            parts = []
            for part in item.get("content", []):
                if part["type"] in {
                    "input_text",
                    "text",
                    "output_text",
                    "audio",
                    "output_audio",
                }:
                    parts.append(part.get("text", part.get("transcript", "")))
                elif part["type"] in {"input_audio", "input_image"}:
                    if "_rows" not in part:
                        part["_rows"] = self.image(part["_bytes"])
                    rows = part["_rows"]
                    marker = f"{{{{frankie_media_{len(media)}}}}}"
                    media.append((marker, rows))
                    parts.append(
                        ("<|vision_start|>" + marker + "<|vision_end|>")
                        if part["type"] == "input_image"
                        else marker
                    )
                    if part["type"] == "input_audio" and part.get("_listener_transcript"):
                        parts.append("Speech transcript (may contain errors): "
                                     + part["_listener_transcript"])
            messages.append({"role": item["role"], "content": "\n".join(parts)})
        tools = [
            {
                "type": "function",
                "function": {k: v for k, v in t.items() if k != "type"},
            }
            for t in settings.get("tools", [])
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=generation_prompt,
            enable_thinking=settings.get("thinking", "off") != "off",
        )
        pad = self.vision_spec.image_token_id
        ids, all_rows, digests, counts = [], [], [], []
        for marker, rows in media:
            before, prompt = prompt.split(marker, 1)
            ids.extend(self.tokenizer.encode(before, add_special_tokens=False))
            ids.extend([pad] * len(rows))
            all_rows.append(rows)
            counts.append(len(rows))
            digests.append(
                int.from_bytes(
                    hashlib.sha256(
                        memoryview(__import__("numpy").asarray(rows.astype(mx.float32)))
                    ).digest()[:8],
                    "little",
                )
            )
        ids.extend(self.tokenizer.encode(prompt, add_special_tokens=False))
        splice = (
            VisionSplice(
                pad,
                mx.concatenate(all_rows),
                image_digests=tuple(digests),
                pad_counts=tuple(counts),
            )
            if media
            else None
        )
        return ids, splice

    def warm(self, settings, *, session_id, bank=None):
        ids, _ = self.prompt(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "warmup"}],
                }
            ],
            settings,
            generation_prompt=False,
        )
        # The checkpoint template requires a user query. Keep only its exact
        # system/tool prefix; the synthetic user never enters the cache.
        start = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        ids = ids[: max(i for i, t in enumerate(ids) if t == start)]
        generate_mtpk(
            self.runtime,
            ids,
            max_tokens=0,
            sampler=SamplerConfig(temperature=0),
            speculative_depth=max(1, self.mtp),
            mtp_history_policy="committed",
            verify_strategy="capture_commit",
            session_bank=self.bank if bank is None else bank,
            session_id=session_id,
            commit_prompt_state_to_bank=True,
        )

    def respond(self, items, settings, emit, abort, *, session_id):
        started = time.monotonic()

        def check_abort():
            if abort.is_set():
                raise InterruptedError("Response cancelled.")

        check_abort()
        self.audio.reset_speech_context()
        ids, splice = self.prompt(items, settings, emit=emit)
        if len(ids) + settings["max_output_tokens"] > settings.get("context", 131072):
            raise ValueError("Conversation exceeds the configured context limit.")
        text_ids = []
        speech_ids = []
        speech_states = []
        pending = deque()
        speaker = None
        all_text = ""
        sent_text = ""
        in_thinking = settings.get("thinking", "off") != "off"
        in_tool = False
        audio_seconds = 0.0
        audio_start = None
        chunk_text = ""
        chunk_start = 0.0
        marks = []
        mouth_enabled = "audio" in settings["output_modalities"]
        detokenizer = copy.copy(self.tokenizer.detokenizer)
        detokenizer.reset()

        def chunk():
            nonlocal speech_ids, speech_states
            text = self.tokenizer.decode(speech_ids).strip()
            if text and re.search(r"\w", text):
                pending.append((text, mx.stack(speech_states)))
            speech_ids = []
            speech_states = []

        def step_audio(force=False):
            nonlocal speaker, audio_seconds, audio_start, chunk_text, chunk_start
            if not mouth_enabled:
                return
            # Keep enough playable audio ahead of the brain, but bound the
            # backlog so an interruption can stop promptly.
            while not abort.is_set():
                now = time.monotonic()
                lead = (
                    audio_seconds - (now - audio_start)
                    if audio_start is not None
                    else 0
                )
                background = getattr(self, "background_step", None)
                target_lead = 0.8 if background is not None else 0.4
                if lead >= target_lead:
                    if background is not None and background(lead):
                        if not force:
                            return
                        continue
                    if not force:
                        return
                    abort.wait(min(0.02, lead - target_lead + 0.05))
                    continue
                if speaker is None:
                    if not pending:
                        return
                    chunk_text, states = pending.popleft()
                    chunk_start = audio_seconds
                    speaker = self.audio.speak(chunk_text, states)
                try:
                    pcm = next(speaker)
                except StopIteration:
                    marks.append(
                        {
                            "text": chunk_text,
                            "start_ms": round(chunk_start * 1000),
                            "end_ms": round(audio_seconds * 1000),
                        }
                    )
                    emit("chunk", marks[-1])
                    speaker = None
                    continue
                if abort.is_set():
                    return
                if audio_start is None:
                    audio_start = time.monotonic()
                audio_seconds += len(pcm) / 24000
                emit("audio", pcm)

        def finish_phrase(value):
            if not mouth_enabled or not speech_ids or in_thinking or in_tool:
                return False
            current = self.tokenizer.decode(speech_ids).strip()
            if value in {"<tool_call>", "<think>"} or (
                value.startswith((" ", "\n"))
                and (
                    re.search(r"[.!?]$", current)
                    or (len(current.split()) >= 4 and re.search(r"[;:,]$", current))
                    or len(current.split()) >= 16
                )
            ):
                chunk()
                return True
            return False

        def received(tokens, states=None):
            nonlocal all_text, sent_text, in_thinking, in_tool
            check_abort()
            for token, state in zip(
                tokens, states if states is not None else repeat(None)
            ):
                value = self.tokenizer.decode([token])
                text_ids.append(token)
                if value == "<think>":
                    # A native reasoning opener ends the public phrase even
                    # when there is no following whitespace token. Speech must
                    # not wait until the private block finishes.
                    if mouth_enabled and not in_thinking and not in_tool:
                        chunk()
                    in_thinking = True
                    continue
                if value == "</think>":
                    in_thinking = False
                    continue
                if in_thinking:
                    continue
                if value == "<tool_call>":
                    chunk()
                    in_tool = True
                    continue
                if value == "</tool_call>":
                    in_tool = False
                    continue
                if in_tool:
                    continue
                finish_phrase(value)
                if mouth_enabled:
                    speech_ids.append(token)
                    speech_states.append(state)
                detokenizer.add_token(token)
                all_text += detokenizer.last_segment
            if all_text != sent_text:
                emit("text", all_text[len(sent_text) :])
                sent_text = all_text
            step_audio()
            if not mouth_enabled:
                background = getattr(self, "background_step", None)
                if background is not None:
                    background(float("inf"))
            check_abort()

        thinking = settings.get("thinking", "off")
        options = {
            "thinking_guard": thinking_guard(self.tokenizer, thinking),
            "max_tokens": settings["max_output_tokens"],
            "sampler": brain_sampler(settings, realtime=True),
            "seed": settings.get("seed", 0),
            "stop_token_ids": set(self.tokenizer.eos_token_ids),
            "abort_check": abort.is_set,
            "vision_splice": splice,
            "session_bank": self.bank,
            "session_id": session_id,
            "session_restore_mode": "clone",
            "capture_final_state": True,
        }
        try:
            # Text consumers need committed tokens, but no speech features or
            # their extra device synchronization and one-forward stream delay.
            feature_stream = (
                CommittedFeatures(self.runtime, len(ids), received)
                if mouth_enabled
                else nullcontext()
            )
            with feature_stream as features:
                def committed(tokens):
                    features.commit(tokens)
                    # A committed next token can finish a fully featured phrase
                    # before that next token has its own hidden-state row.
                    if features.emitted < len(features.tokens) and finish_phrase(
                        self.tokenizer.decode([features.tokens[features.emitted]])
                    ):
                        step_audio()
                        check_abort()

                callback = committed if features is not None else received
                if self.mtp:
                    result = generate_mtpk(
                        self.runtime,
                        ids,
                        speculative_depth=self.mtp,
                        mtp_history_policy="committed",
                        verify_strategy="capture_commit",
                        token_callback=callback,
                        commit_prompt_state_to_bank=True,
                        **options,
                    )
                else:
                    result = generate_ar(
                        self.runtime, ids, token_callback=callback, **options
                    )
                if features is not None and not abort.is_set():
                    features.flush(final=True)
            if not abort.is_set():
                detokenizer.finalize()
                tail = detokenizer.last_segment
                if tail:
                    all_text += tail
                    emit("text", tail)
                chunk()
                step_audio(force=True)
            return {
                "text": all_text,
                "raw_text": self.tokenizer.decode(text_ids),
                "stats": result.stats.to_dict(),
                "finish_reason": result.finish_reason,
                "audio_seconds": audio_seconds,
                "chunks": marks,
                "seconds": time.monotonic() - started,
            }
        finally:
            if speaker is not None:
                speaker.close()
            self.audio.reset_speech_context()
