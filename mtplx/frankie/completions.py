"""Shared-brain HTTP experiment. All MLX work stays on Frankie's model worker.

The batch generator owns only request caches. Voice yields at audio lead
boundaries; an idle executor advances the same bounded steps between turns.
"""

import asyncio
import copy
import contextvars
import hmac
import json
import os
import queue
import threading
import time
import uuid
from collections import deque

from .sampling import brain_sampler, thinking_guard, thinking_mode


class Job:
    def __init__(self, data, chat, loop):
        self.data, self.chat, self.loop = data, chat, loop
        self.id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        self.created = int(time.time())
        self.events = queue.Queue(maxsize=128)
        self.ready = asyncio.Event()
        self.cancelled = threading.Event()
        self.uid = None
        self.ids = None
        self.splice = None
        self.cache = None
        self.offset = 0
        self.tokens = []
        self.raw = ""
        self.content = ""
        self.reasoning = ""
        self.done = False
        self.steps = None
        self.context = None
        self.prefill_step_size = 64
        self.cached_tokens = 0
        self.internal = False
        self.prepare_callback = None
        self.prepare_only = False
        self.bank = None

    def emit(self, value):
        try:
            self.events.put_nowait(value)
        except queue.Full:
            self.cancelled.set()
        self.loop.call_soon_threadsafe(self.ready.set)

    async def receive(self):
        while True:
            self.ready.clear()
            try:
                return self.events.get_nowait()
            except queue.Empty:
                if self.cancelled.is_set():
                    raise RuntimeError("Completion consumer stopped keeping up.")
                await self.ready.wait()


class Completions:
    def __init__(self, engine, executor, slots=4, context_tokens=4096):
        self.engine, self.executor = engine, executor
        self.slots, self.context_tokens = slots, context_tokens
        self.loop = asyncio.get_running_loop()
        self.jobs = set()
        self.pending = deque()
        self.active = {}
        self.batch = None
        self.driver = None
        self.listener_bank = None

    def submit(self, data, chat, *, internal=False, prepare=None, prepare_only=False):
        if prepare_only and (not internal or not callable(prepare)):
            raise ValueError("Prepare-only work requires an internal owner callback.")
        # A bounded listener request shares the model owner, but cannot be
        # starved by client HTTP admission. It never duplicates model weights.
        jobs = tuple(self.jobs)  # The inference owner can retire jobs concurrently.
        if internal and any(j.internal for j in jobs):
            raise OverflowError("A listener observation is already in flight.")
        if not internal and sum(not j.internal for j in jobs) >= self.slots:
            raise OverflowError("All HTTP slots are busy.")
        job = Job(data, chat, self.loop)
        job.internal = internal
        job.prepare_callback = prepare
        job.prepare_only = prepare_only
        self.jobs.add(job)
        (self.pending.appendleft if internal else self.pending.append)(job)
        self.engine.background_step = self.voice_step
        if self.driver is None or self.driver.done():
            self.driver = asyncio.create_task(self.drive())
        return job

    def submit_prepare(self, callback):
        """One bounded internal owner operation, with no brain warmup/prefill."""
        return self.submit({}, True, internal=True, prepare=callback, prepare_only=True)

    def run_preparation(self, job):
        try:
            if not job.cancelled.is_set():
                started = time.monotonic()
                value = job.prepare_callback(job)
                if not job.cancelled.is_set():
                    job.done = True
                    job.emit({"prepared": value,
                              "listener_prepare_seconds": time.monotonic() - started})
        except Exception as exc:  # noqa: BLE001 — owner callback errors cross this job boundary.
            if not job.cancelled.is_set():
                job.emit({"error": str(exc)})
        finally:
            job.prepare_callback = None
            self.jobs.discard(job)

    async def drive(self):
        while self.jobs:
            try:
                await self.loop.run_in_executor(self.executor, self.step)
            except Exception as exc:
                for job in list(self.jobs):
                    job.emit({"error": str(exc)})
                self.jobs.clear()
                raise
            await asyncio.sleep(0)

    def voice_step(self, lead):
        # Do not compete with the first spoken chunk. Keep a conservative
        # playback reserve, and let the existing audio producer refill it.
        if lead < 0.6:
            return False
        self.step(voice=True)
        return True

    def warm_listener(self):
        """Warm the immutable instruction prefix on the inference owner.

        Call through ``executor`` before enabling semantic overlap. Every
        observation clones this exact state; it never becomes the voice cache
        or gets replaced by a particular user's classifier input/output.
        """
        if self.listener_bank is None:
            from mtplx.session_bank import SessionBank
            self.listener_bank = SessionBank(
                max_entries=1, max_bytes=512 * 1024**2,
                per_session_max_bytes=512 * 1024**2,
                # Static instructions live as long as these loaded weights.
                # Expiry must not trigger a cold warmup during live overlap.
                idle_ttl_s=float("inf"),
            )
        if self.engine.mtp and not len(self.listener_bank):
            from .interaction import ListenerObservation, semantic_messages
            instructions = semantic_messages(
                ListenerObservation("warmup", 0, 0, ""), compact=True,
            )[0]["content"]
            self.engine.warm(
                {"instructions": instructions, "thinking": "off", "tools": []},
                session_id="listener-prefix", bank=self.listener_bank,
            )
            if not len(self.listener_bank):
                raise RuntimeError("Listener instruction cache exceeded its memory budget.")
        return {"cache_bytes": self.listener_bank.total_nbytes}

    def prepare(self, job):
        from mtplx.server.omlx_bridge.thinking import ThinkingParser
        from mtplx.server.omlx_bridge.tool_calling import ToolCallStreamFilter

        if job.internal:
            self.warm_listener()
            job.bank = self.listener_bank
        else:
            job.bank = self.engine.bank
        if job.prepare_callback is not None:
            # Includes acoustic inference, if needed, on the same owner as
            # brain/mouth inference and with voice feature capture suspended.
            started = time.monotonic()
            job.prepare_callback(job)
            job.emit({"listener_prepare_seconds": time.monotonic() - started})
            job.prepare_callback = None
        data = job.data
        if job.chat:
            items, instructions = [], []
            for message in data["messages"]:
                role = message["role"]
                content = message.get("content") or ""
                parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else []
                if isinstance(content, list):
                    for part in content:
                        if part["type"] == "text":
                            parts.append({"type": "input_text", "text": part["text"]})
                        elif part["type"] == "image_url":
                            parts.append({"type": "input_image", "_bytes": part["_bytes"]})
                        else:
                            raise ValueError("Unsupported chat content type.")
                if role in {"system", "developer"}:
                    instructions.extend(p["text"] for p in parts)
                elif role == "tool":
                    items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": "\n".join(p["text"] for p in parts)})
                else:
                    items.append({"type": "message", "role": role, "content": parts})
                    for call in message.get("tool_calls") or []:
                        items.append({"type": "function_call", "call_id": call["id"], **call["function"]})
            job.ids, job.splice = self.engine.prompt(items, {
                "instructions": "\n\n".join(instructions),
                "thinking": data["thinking"],
                "tools": [{"type": "function", **t["function"]} for t in data.get("tools") or []],
            })
        else:
            prompt = data["prompt"]
            job.ids = self.engine.tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
        if not job.ids or len(job.ids) + data["max_tokens"] > self.context_tokens:
            raise ValueError(f"Prompt and output must fit within {self.context_tokens} tokens.")
        if job.splice is not None and not self.engine.mtp:
            from mlx_lm.models.cache import make_prompt_cache
            job.cache = make_prompt_cache(self.engine.runtime.model)
        job.detokenizer = copy.copy(self.engine.tokenizer.detokenizer)
        job.detokenizer.reset()
        job.thinking = ThinkingParser(starts_in_thinking=bool(data.get("enable_thinking")))
        job.tools = ToolCallStreamFilter(self.engine.tokenizer)
        if self.engine.mtp:
            return
        import mlx.core as mx
        from mlx_lm.sample_utils import apply_top_k, apply_top_p, make_logits_processors
        config = brain_sampler(data)
        processors = make_logits_processors(presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty, presence_context_size=data["max_tokens"],
            frequency_context_size=data["max_tokens"])
        job.processors = [lambda tokens, logits, p=p: p(tokens[len(job.ids):], logits) for p in processors]
        guard_config = thinking_guard(self.engine.tokenizer, data["thinking"])
        if guard_config is not None:
            from mtplx.thinking_guard import ThinkingGuard
            guard = ThinkingGuard(guard_config)
            def bound_reasoning(tokens, logits):
                generated = tokens[len(job.ids):].tolist()
                guard.observe(generated)
                overlay = guard.overlay_for(generated) if guard.steering_active else None
                if overlay:
                    logits = logits.at[:, mx.array(list(overlay))].add(-mx.array(list(overlay.values())))
                return logits
            job.processors.append(bound_reasoning)
        # Explicit keys leave the voice and mouth's random stream untouched.
        key = mx.random.key(data.get("seed", int(uuid.uuid4().hex[:8], 16)))
        def sample(logprobs):
            nonlocal key
            temperature = config.temperature
            if temperature == 0:
                return mx.argmax(logprobs, axis=-1)
            if temperature != 1:
                logprobs = logprobs / temperature
                logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
            if 0 < config.top_p < 1:
                logprobs = apply_top_p(logprobs, config.top_p)
            if 0 < config.top_k < logprobs.shape[-1]:
                logprobs = apply_top_k(logprobs, config.top_k)
            key, draw = mx.random.split(key)
            return mx.random.categorical(logprobs, key=draw)
        job.sampler = sample

    def insert(self, job):
        if self.engine.mtp:
            # Admission can run inside a voice callback. Start with defaults
            # so a text request cannot inherit that conversation's image rope.
            job.context = contextvars.Context()
            job.steps = self.mtp_steps(job)
            job.uid = job.id
            self.active[job.uid] = job
            return
        if self.batch is None:
            from mlx_lm.generate import BatchGenerator
            self.batch = BatchGenerator(self.engine.runtime.model, completion_batch_size=self.slots,
                prefill_batch_size=2, prefill_step_size=16,
                stop_tokens=[[t] for t in self.engine.tokenizer.eos_token_ids])
        job.uid = self.batch.insert([job.ids[job.offset:]], max_tokens=[job.data["max_tokens"]],
            caches=[job.cache] if job.cache else None, all_tokens=[job.ids[:job.offset]],
            samplers=[job.sampler], logits_processors=[job.processors])[0]
        self.active[job.uid] = job
        job.cache = job.splice = None

    def mtp_steps(self, job):
        # Use the same prefill, draft/verify, committed-history and cache-repair
        # code as voice. Only the caches and RNG belong to this HTTP request.
        from mtplx.generation import (
            generate_mtpk, restore_or_prefill_prompt_state, _vision_rope_scope_for,
        )
        runtime = self.engine.runtime
        bank = getattr(job, "bank", None)
        if bank is None:
            bank = self.engine.bank
        with _vision_rope_scope_for(job.splice):
            state = yield from restore_or_prefill_prompt_state.steps(
                runtime, job.ids,
                mtp_history_policy="committed", session_bank=bank,
                restore_mode="clone", session_id=job.id,
                store_prefix_snapshot=False,
                abort_check=job.cancelled.is_set, vision_splice=job.splice,
                prefill_chunk_size=64,
                prefill_step_size=lambda: job.prefill_step_size,
            )
            job.cached_tokens = state.cached_tokens
            def received(tokens):
                if job.cancelled.is_set():
                    raise InterruptedError("Completion cancelled.")
                for token in tokens:
                    job.tokens.append(token)
                    job.detokenizer.add_token(token)
                    self.emit_text(job, job.detokenizer.last_segment)
            result = yield from generate_mtpk.steps(
                runtime, job.ids, _prompt_state=state,
                session_bank=bank, session_id=job.id,
                session_restore_mode="clone",
                commit_prompt_state_to_bank=not getattr(job, "internal", False),
                speculative_depth=self.engine.mtp, mtp_history_policy="committed",
                verify_strategy="capture_commit", max_tokens=job.data["max_tokens"],
                sampler=brain_sampler(job.data),
                thinking_guard=thinking_guard(self.engine.tokenizer, job.data["thinking"]),
                seed=job.data.get("seed", int(uuid.uuid4().hex[:8], 16)),
                stop_token_ids=set(self.engine.tokenizer.eos_token_ids),
                token_callback=received, abort_check=job.cancelled.is_set,
                vision_splice=job.splice,
            )
        job.tokens = result.tokens
        if os.environ.get("FRANKIE_HTTP_PROFILE"):
            print("http_mtp", json.dumps({"id": job.id, "depth": self.engine.mtp,
                "prompt_tokens": len(job.ids), "generated_tokens": len(result.tokens),
                "cached_tokens": job.cached_tokens,
                "drafted_tokens": result.stats.drafted_tokens,
                "accepted_drafts": result.stats.accepted_drafts,
                "prefill_seconds": result.stats.prompt_eval_time_s}), flush=True)
        return result.finish_reason

    def close_mtp_job(self, job):
        if job.steps is not None:
            job.context.run(job.steps.close)
            job.steps = job.context = None
        job.cache = job.splice = None
        self.active.pop(job.uid, None)
        self.jobs.discard(job)

    def step_mtp(self, voice):
        for job in list(self.active.values()):
            if job.cancelled.is_set():
                self.close_mtp_job(job)
        if self.pending:
            job = self.pending.popleft()
            if job.cancelled.is_set():
                self.jobs.discard(job)
            else:
                try:
                    self.prepare(job)
                    self.insert(job)
                except Exception as exc:
                    self.jobs.discard(job)
                    job.emit({"error": str(exc)})
            return
        if self.active:
            # Only one bounded internal probe is admitted. Prefer its short
            # decoding slices, yielding back to speech between every slice.
            uid = next((uid for uid, j in self.active.items() if getattr(j, "internal", False)),
                       next(iter(self.active)))
            job = self.active.pop(uid)
            self.active[uid] = job
            try:
                # Batch idle prefill efficiently, but return to a small slice
                # as soon as voice needs the shared inference owner. Yield
                # after each slice so a new voice turn can start promptly.
                job.prefill_step_size = 32 if voice else 64
                # This legacy layout hint is process-global. Restore the voice
                # owner's value when yielding between independent HTTP jobs.
                context_key = "MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"
                previous_context = os.environ.get(context_key)
                os.environ[context_key] = str(len(job.ids))
                try:
                    job.context.run(next, job.steps)
                finally:
                    if previous_context is None:
                        os.environ.pop(context_key, None)
                    else:
                        os.environ[context_key] = previous_context
            except StopIteration as done:
                self.close_mtp_job(job)
                self.finish(job, done.value)
            except Exception as exc:
                self.close_mtp_job(job)
                if not job.cancelled.is_set():
                    job.emit({"error": str(exc)})

    def emit_text(self, job, text, final=False):
        job.raw += text
        if not job.chat:
            if text:
                job.content += text
                job.emit({"text": text})
            return
        reasoning, content = job.thinking.feed(text)
        if final:
            r, c = job.thinking.finish()
            reasoning += r
            content += c
        content = job.tools.feed(content)
        if final:
            content += job.tools.finish()
        delta = {}
        if content:
            job.content += content
            delta["content"] = content
        if reasoning:
            job.reasoning += reasoning
            delta["reasoning_content"] = reasoning
        if delta:
            job.emit(delta)

    def finish(self, job, reason):
        from .thinking import public_tool_calls
        job.detokenizer.finalize()
        self.emit_text(job, job.detokenizer.last_segment, final=True)
        calls = public_tool_calls(job.raw, self.engine.tokenizer, job.data.get("tools"),
                                  starts_in_thinking=bool(job.data.get("enable_thinking"))) if job.chat else None
        if calls:
            job.emit({"tool_calls": [{"index": i, **call} for i, call in enumerate(calls)]})
            reason = "tool_calls"
        result = {"content": job.content}
        if job.reasoning:
            result["reasoning_content"] = job.reasoning
        if calls:
            result["tool_calls"] = calls
        job.done = True
        job.emit({"finish_reason": reason, "message": result,
            "usage": {"prompt_tokens": len(job.ids), "completion_tokens": len(job.tokens),
                      "prompt_tokens_details": {"cached_tokens": job.cached_tokens},
                      "total_tokens": len(job.ids) + len(job.tokens)}})
        self.jobs.discard(job)

    def step(self, *, voice=False):
        import mlx.core as mx
        model = getattr(self.engine.runtime.model, "language_model", self.engine.runtime.model)
        features = getattr(model, "_mtplx_feature_stream", None)
        model._mtplx_feature_stream = None
        try:
            if self.pending and getattr(self.pending[0], "prepare_only", False):
                self.run_preparation(self.pending.popleft())
                if not self.jobs:
                    if self.batch is not None:
                        self.batch.close()
                        self.batch = None
                    self.engine.background_step = None
                return
            if self.engine.mtp:
                self.step_mtp(voice)
                if not self.jobs:
                    self.engine.background_step = None
                return
            for uid, job in list(self.active.items()):
                if job.cancelled.is_set():
                    self.batch.remove([uid])
                    self.active.pop(uid)
                    self.jobs.discard(job)
            if self.pending:
                job = self.pending[0]
                if job.cancelled.is_set():
                    self.pending.popleft()
                    self.jobs.discard(job)
                else:
                    try:
                        if job.ids is None:
                            self.prepare(job)
                        elif job.splice is not None and job.offset < len(job.ids) - 1:
                            from mtplx.vision.splice import spliced_chunk_embeddings
                            end = min(job.offset + (32 if voice else 128), len(job.ids) - 1)
                            ids = mx.array([job.ids[job.offset:end]])
                            rows = spliced_chunk_embeddings(self.engine.runtime.embed_tokens, ids, job.splice)
                            self.engine.runtime.forward_ar(ids, cache=job.cache, input_embeddings=rows, emit_logits=False)
                            mx.eval([c.state for c in job.cache])
                            job.offset = end
                            if os.environ.get("FRANKIE_HTTP_PROFILE"):
                                print("http_image_prefill", json.dumps({"at": time.monotonic(),
                                    "id": job.id, "offset": end, "total": len(job.ids)}), flush=True)
                        else:
                            self.insert(job)
                            self.pending.popleft()
                    except Exception as exc:
                        self.pending.popleft()
                        self.jobs.discard(job)
                        job.emit({"error": str(exc)})
                if self.pending and self.pending[0] is job:
                    self.pending.rotate(-1)
            if self.active:
                # mlx-lm copies the step size into its active prompt batch.
                size = 32 if voice else 128
                self.batch.prefill_step_size = size
                self.batch._prompt_batch.prefill_step_size = size
                prefills, generated = self.batch.next()
                if os.environ.get("FRANKIE_HTTP_PROFILE") and prefills:
                    print("http_prefill", json.dumps({"at": time.monotonic(), "rows": [
                        {"uid": p.uid, "progress": p.progress} for p in prefills]}), flush=True)
                for response in generated:
                    job = self.active[response.uid]
                    job.tokens.append(int(response.token))
                    if response.finish_reason != "stop":
                        job.detokenizer.add_token(int(response.token))
                        self.emit_text(job, job.detokenizer.last_segment)
                    if response.finish_reason:
                        self.active.pop(response.uid)
                        self.finish(job, response.finish_reason)
            if not self.jobs:
                if self.batch is not None:
                    self.batch.close()
                    self.batch = None
                self.engine.background_step = None
        except Exception as exc:
            for job in list(self.jobs):
                job.emit({"error": str(exc)})
            self.jobs.clear()
            self.pending.clear()
            self.active.clear()
            if self.batch is not None:
                self.batch.close()
                self.batch = None
            self.engine.background_step = None
        finally:
            model._mtplx_feature_stream = features


def attach_routes(app, get_engine, executor, token, *, slots=4, context_tokens=4096):
    from fastapi import Request
    from fastapi.responses import JSONResponse, StreamingResponse
    service = None

    def get_service():
        nonlocal service
        if service is None:
            service = Completions(get_engine(), executor, slots, context_tokens)
        return service

    async def complete(request: Request):
        nonlocal service
        if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
            return JSONResponse({"error": {"message": "Invalid access token."}}, status_code=401)
        try:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 18 * 1024**2:
                    raise ValueError("Request body exceeds 18 MiB.")
            data = json.loads(body)
            chat = request.url.path.endswith("/chat/completions")
            # Reuse the ordinary server's request types without importing its
            # inference service on the voice-only startup path.
            from mtplx.server.completion_requests import ChatCompletionRequest, CompletionRequest
            schema = ChatCompletionRequest if chat else CompletionRequest
            data = schema.model_validate(data).model_dump(exclude_none=True)
            data["thinking"] = thinking_mode(data) if chat else "off"
            data["enable_thinking"] = data["thinking"] != "off"
            brain_sampler(data)
            limit = data.get("max_completion_tokens", data.get("max_tokens", 512))
            if not 1 <= limit <= min(32768, context_tokens):
                raise ValueError(f"max_tokens must be between 1 and {min(32768, context_tokens)}.")
            data["max_tokens"] = limit
            # Common clients send these explicit default values.
            if data.get("tool_choice") in ("auto", "none"):
                if data["tool_choice"] == "none":
                    data["tools"] = []
                data.pop("tool_choice")
            if data.get("stop") == []:
                data.pop("stop")
            if data.get("response_format") == {"type": "text"}:
                data.pop("response_format")
            for key in ("stop", "logprobs", "top_logprobs", "response_format", "tool_choice"):
                if data.get(key) is not None:
                    raise ValueError(f"{key} is not wired into the shared-brain experiment yet.")
            if data.get("n", 1) != 1 or data.get("echo", False):
                raise ValueError("Use independent concurrent requests; n and echo are not wired yet.")
            if not chat and not (isinstance(data.get("prompt"), str) or isinstance(data.get("prompt"), list) and all(type(t) is int for t in data["prompt"])):
                raise ValueError("prompt must be text or one list of token ids.")
            # Fetch media outside the MLX owner so a slow URL cannot block voice.
            if chat:
                from mtplx.vision.media import image_bytes_from_url, validate_image_detail
                count = 0
                for message in data["messages"]:
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if not isinstance(part, dict) or part.get("type") not in ("text", "image_url"):
                            raise ValueError("Use text or image_url content parts.")
                        if part["type"] == "text":
                            if not isinstance(part.get("text"), str):
                                raise ValueError("Text content must be a string.")
                            continue
                        if message["role"] != "user":
                            raise ValueError("Images belong in user messages.")
                        count += 1
                        if count > 4:
                            raise ValueError("At most four images per request.")
                        image = part["image_url"]
                        if not isinstance(image, dict):
                            raise ValueError("image_url must be an object containing url.")
                        validate_image_detail(image.get("detail", "auto"))
                        part["_bytes"] = await asyncio.to_thread(
                            image_bytes_from_url, image["url"], max_bytes=12 * 1024**2)
            job = get_service().submit(data, chat)
        except OverflowError as exc:
            return JSONResponse({"error": {"message": str(exc), "type": "rate_limit_error"}}, status_code=429)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error"}}, status_code=400)

        def envelope(choice, *, streaming=True, usage=None):
            value = {"id": job.id, "created": job.created, "model": data.get("model", "Frankie"),
                     "object": "chat.completion.chunk" if chat and streaming else "chat.completion" if chat else "text_completion",
                     "choices": [{"index": 0, **choice}]}
            if usage is not None:
                value["usage"] = usage
            return value

        async def stream():
            try:
                if chat:
                    yield "data: " + json.dumps(envelope({"delta": {"role": "assistant"}, "finish_reason": None})) + "\n\n"
                while True:
                    event = await job.receive()
                    if "error" in event:
                        yield "data: " + json.dumps({"error": {"message": event["error"]}}) + "\n\n"
                        break
                    finish = event.get("finish_reason")
                    delta = {} if finish else event
                    choice = {"delta": delta, "finish_reason": finish} if chat else {"text": delta.get("text", ""), "finish_reason": finish}
                    yield "data: " + json.dumps(envelope(choice)) + "\n\n"
                    if finish:
                        if data.get("stream_options", {}).get("include_usage"):
                            yield "data: " + json.dumps({**envelope({}, usage=event["usage"]), "choices": []}) + "\n\n"
                        break
                yield "data: [DONE]\n\n"
            finally:
                job.cancelled.set()

        if data["stream"]:
            return StreamingResponse(stream(), media_type="text/event-stream")
        try:
            while True:
                if await request.is_disconnected():
                    return JSONResponse({"error": {"message": "Client disconnected."}}, status_code=499)
                try:
                    event = await asyncio.wait_for(job.receive(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if "error" in event:
                    return JSONResponse({"error": {"message": event["error"]}}, status_code=400)
                if "finish_reason" in event:
                    choice = {"message": {"role": "assistant", **event["message"]}} if chat else {"text": event["message"]["content"]}
                    return envelope({**choice, "finish_reason": event["finish_reason"]}, streaming=False, usage=event["usage"])
        finally:
            job.cancelled.set()

    @app.get("/v1/models")
    async def models(request: Request):
        if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
            return JSONResponse({"error": {"message": "Invalid access token."}}, status_code=401)
        return {"object": "list", "data": [
            {"id": "Frankie", "object": "model", "created": 0, "owned_by": "local"},
        ]}

    app.add_api_route("/v1/chat/completions", complete, methods=["POST"])
    app.add_api_route("/v1/completions", complete, methods=["POST"])
    return get_service
