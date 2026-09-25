"""One inference owner for brain, vision, learned audio input and speech output."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path

import mlx.core as mx
import numpy as np

from mtplx.features import CommittedFeatures
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank
from mtplx.vision import load_vision_tower, vision_spec_for_model_dir
from mtplx.vision.processing import decode_image, preprocess_images
from mtplx.vision.splice import VisionSplice

from .audio import AudioModels
from .detokenizing import new_detokenizer
from .interruption import REGENERATION_INSTRUCTIONS, draft_notice
from .sampling import brain_sampler, thinking_guard
from .thinking import public_tool_calls


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

    def prompt(self, items, settings, *, generation_prompt=True, emit=None, public_history=False):
        instructions = settings["instructions"]
        if settings.get("streaming_listener", "off") != "off" or any(
                item.get("_playback_interrupted") or "_interrupted_draft" in item for item in items):
            instructions += "\n\n" + REGENERATION_INSTRUCTIONS
        messages = [{"role": "system", "content": instructions}]
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
                    # Prepared audio/image rows are immutable. Keep their cache
                    # identity with the rows, not a request or positional index.
                    identity = part.get("_rows_identity")
                    if identity is None or identity[0] is not rows:
                        digest = hashlib.sha256(
                            memoryview(np.asarray(rows.astype(mx.float32)))
                        ).digest()
                        identity = part["_rows_identity"] = (
                            rows, int.from_bytes(digest[:8], "little")
                        )
                    media.append((marker, rows, identity[1]))
                    parts.append(
                        ("<|vision_start|>" + marker + "<|vision_end|>")
                        if part["type"] == "input_image"
                        else marker
                    )
                    if part["type"] == "input_audio":
                        # The fast listener skips the final semantic pass that
                        # otherwise supplies grounding. Reuse full-input CTC;
                        # keep the neural audio and never rerun the ear for text.
                        transcript = part.get("_listener_transcript") or (
                            part.get("_transcript", "")
                            if settings.get("streaming_listener") == "backchannel"
                            else ""
                        )
                        if transcript.strip():
                            parts.append("Speech transcript (may contain errors): " + transcript)
            role = "user" if item.get("_task_notice") else item["role"]
            content = "\n".join(parts)
            # An unheard/cancelled reply is not an empty assistant example.
            # Preserve the authoritative item and its playback notice below;
            # real function calls have their own substantive branch above.
            if role != "assistant" or content.strip():
                messages.append({"role": role, "content": content})
            if item.get("_playback_interrupted") or item.get("_interrupted_draft") is not None:
                messages.append({"role": "user", "content": draft_notice(
                    item.get("_interrupted_draft"), has_heard_text=bool(content.strip()))})
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
            # Realtime history contains public speech, not saved reasoning.
            # Do not synthesize empty thinking blocks for completed turns.
            **({"preserve_thinking": False} if public_history else {}),
        )
        pad = self.vision_spec.image_token_id
        ids, all_rows, digests, counts = [], [], [], []
        for marker, rows, digest in media:
            before, prompt = prompt.split(marker, 1)
            ids.extend(self.tokenizer.encode(before, add_special_tokens=False))
            ids.extend([pad] * len(rows))
            all_rows.append(rows)
            counts.append(len(rows))
            digests.append(digest)
        ids.extend(self.tokenizer.encode(prompt, add_special_tokens=False))
        splice = (
            VisionSplice(
                pad,
                all_rows[0] if len(all_rows) == 1 else mx.concatenate(all_rows),
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
        self.cache_prompt(ids, None, session_id=session_id, bank=bank)

    def cache_prompt(self, ids, splice, *, session_id, bank=None, abort_check=None):
        return generate_mtpk(
            self.runtime,
            ids,
            max_tokens=0,
            sampler=SamplerConfig(temperature=0),
            speculative_depth=max(1, self.mtp),
            mtp_history_policy="committed" if self.mtp else "cycle",
            verify_strategy="capture_commit",
            session_bank=self.bank if bank is None else bank,
            session_id=session_id,
            commit_prompt_state_to_bank=True,
            vision_splice=splice,
            abort_check=abort_check,
        )

    def respond(self, items, settings, emit, abort, *, session_id):
        started = time.monotonic()

        def check_abort():
            if abort.is_set():
                raise InterruptedError("Response cancelled.")

        check_abort()
        self.audio.reset_speech_context()
        ids, splice = self.prompt(items, settings, emit=emit, public_history=True)
        if len(ids) + settings["max_output_tokens"] > settings.get("context", 131072):
            raise ValueError("Conversation exceeds the configured context limit.")
        history_prefill = None
        if self.bank is not None:
            # The open thinking header changes when this becomes public history.
            # Bank the closed history instead, so the next turn can reuse every
            # completed audio span without restoring across a rewritten header.
            history_ids, history_splice = self.prompt(
                items, settings, generation_prompt=False, public_history=True
            )
            history_prefill = self.cache_prompt(
                history_ids, history_splice, session_id=session_id, abort_check=abort.is_set
            ).stats.to_dict()
        text_ids = []
        speech_ids = []
        speech_states = []
        pending = deque()
        speaker = None
        all_text = ""
        sent_text = ""
        in_thinking = settings.get("thinking", "off") != "off"
        in_tool = False
        tool_start = None
        native_tools = False
        tool_events = []
        audio_seconds = 0.0
        audio_start = None
        chunk_text = ""
        chunk_start = 0.0
        marks = []
        mouth_enabled = "audio" in settings["output_modalities"]
        detokenizer = new_detokenizer(self)

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
                urgent = getattr(self, "urgent_background_step", None)
                if urgent is not None and urgent(lead):
                    # Only bounded prefix CTC can run here. Recompute reserve
                    # and cancellation before mouth or ordinary background work.
                    continue
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

        def flush_text():
            nonlocal sent_text
            if all_text != sent_text:
                emit("text", all_text[len(sent_text) :])
                sent_text = all_text

        def received(tokens, states=None):
            nonlocal all_text, in_thinking, in_tool, tool_start, native_tools
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
                    # Only committed public native envelopes are eligible.
                    # Nested envelopes stay invalid rather than salvaging one.
                    tool_start = None if in_tool else len(text_ids) - 1
                    native_tools = True
                    in_tool = True
                    continue
                if value == "</tool_call>":
                    if in_tool and tool_start is not None and settings.get("background_tasks"):
                        raw = self.tokenizer.decode(text_ids[tool_start:])
                        for index, call in enumerate(public_tool_calls(
                            raw, self.tokenizer, settings.get("tools", [])
                        )):
                            event = {"key": (len(text_ids), index), "call": call}
                            tool_events.append(event)
                            flush_text()
                            emit("tool_call", event)
                    tool_start = None
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
            flush_text()
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
                        commit_prompt_state_to_bank=False,
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
            stats = result.stats.to_dict()
            if history_prefill is not None:
                stats["history_prefill"] = {key: history_prefill.get(key) for key in (
                    "cached_tokens", "new_prefill_tokens", "session_cache_hit",
                    "prompt_eval_time_s", "elapsed_s")}
            return {
                "text": all_text,
                "raw_text": self.tokenizer.decode(text_ids),
                "stats": stats,
                "finish_reason": result.finish_reason,
                "audio_seconds": audio_seconds,
                "chunks": marks,
                "seconds": time.monotonic() - started,
                # Reuse the same parsed identities at response completion.
                # Other backends/envelopes retain the existing terminal parser.
                **({"tool_events": tool_events}
                   if native_tools and settings.get("background_tasks") else {}),
            }
        finally:
            if speaker is not None:
                speaker.close()
            self.audio.reset_speech_context()
