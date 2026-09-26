# Frankie in one MTPLX server

Frankie accepts text, images, and live microphone audio and streams text and
speech back. One `mtplx frankie` process owns the brain, vision tower, Parakeet
encoder, learned audio bridges, Qwen3-TTS or Breeze, expression conditioning,
VAD, and optional learned turn projection.
There are no model subprocesses or requests to another inference server. The
browser and an optional agent harness are clients of this process.

Powered by [MTPLX](https://github.com/youssofal/MTPLX). Brain inference, MTP,
KV caches, vision, and tool parsing reuse MTPLX. Audio inference reuses
`parakeet-mlx` and `mlx-audio`; the small trained adapters come from Frankie.
The browser AudioWorklet is adapted from Buzz under Apache-2.0.

## Install on Apple Silicon

Python 3.12 and an Apple Silicon Mac are required for this tested path. The
implementation was exercised on an M3 Ultra with 256 GiB unified memory. Model
conversion needs considerably more memory and disk than serving the result.
A full 128K conversation has not been validated in this combined server; do
not assume that the short-conversation footprint establishes long-context fit.

```sh
git clone --branch frankie/realtime https://github.com/tlongwell-block/MTPLX.git
cd MTPLX
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[frankie,dev]'
```

Dependencies for the exercised audio implementation are pinned to
`mlx-audio==0.5.1` and `parakeet-mlx==0.5.2`. MLX 0.32.2 and mlx-lm 0.31.3
were used. Keep the upstream LICENSE and NOTICE files with distributions.

## Prepare the model folders

If you already have the prepared native Frankie folders, skip conversion and
point `--brain` and `--audio` at them. Keep their manifests together when sharing
the native package. No recording needs to be distributed alongside them.

For a new build, obtain these inputs:

- The complete Frankie GGUF package, supplied separately. Its learned ear,
  tone, VAD, expression, and default voice conditioning are reused.
- The source Hugging Face checkpoint for the matching 27B brain, including its
  own MTP weights, tokenizer, and vision tower. This port was tested with
  `huihui-ai/Huihui-Qwen3.8-27B-abliterated` revision
  `739e3c5b89849f6c238ce1e5b70008612ae42cdd`.
- [Parakeet's MLX source weights](https://huggingface.co/mlx-community/parakeet-tdt_ctc-110m).
- [Qwen3-TTS Base source weights](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base),
  including `speech_tokenizer/`. Use the Base model with speaker conditioning.

Download complete source snapshots to local folders, then set the paths below.
Choose new output directories; preparation refuses to overwrite an existing
audio package.

```sh
brain_source='/absolute/path/to/brain-source'
ear_source='/absolute/path/to/parakeet-source'
mouth_source='/absolute/path/to/qwen-tts-source'
frankie_gguf='/absolute/path/to/frankie.gguf'
export MTPLX_FORGE_MODEL_ROOT="$PWD/models"
export MTPLX_CONFIG="$PWD/frankie-config.toml"
export HF_HOME="$PWD/hf-cache"
export TOKENIZERS_PARALLELISM=false

mtplx forge build \
  --repo "$brain_source" --out "$PWD/forge" --run-id frankie-q4 \
  --recipe '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}' \
  --branded-name Frankie-brain-q4-mtp --max-tokens 128 --json

python -m mtplx.frankie.prepare \
  --gguf "$frankie_gguf" --ear "$ear_source" --mouth "$mouth_source" \
  --output "$PWD/models/Frankie-audio-q8"
```

Use the model directory reported by Forge as `--brain`. Forge converts the
brain and restores its vision and MTP components using existing upstream code.
The audio preparer converts eligible linear and embedding weights to 8-bit
with group size 64; the speech codec and other layers retain source precision.
The brain uses MLX 4-bit affine quantization, with its MTP head in BF16. This is
not a byte-identical quantization of the GGUF's brain tensors.

The preparer reads the packaged reference recording only in memory to obtain a
speaker embedding. It saves conditioning tensors, reference codec tokens, and
reference text token IDs, never a standalone reference WAV. It imports the
trained expression directions exactly, including their existing scaling.

Audio preparation requires the combined `assets.turn.gguf` VAP-BC asset. Build
it with the llama.cpp fork's `convert-turn.py` and include it with `pack.py --turn`.
The verified source is the MIT `maai-kyoto/vap_bc_en`
checkpoint, revision `3ff203ce14de279045eb1b145a1dd24caa8f1c9d`, SHA256
`54e5d19456c0ec7a6fb54ebbbceca837257d827bc86fd7820b0669f961cd11bc`.
Preparation imports its own VAP objective and BC head under `turn.*`, sharing
one streaming encoder/attention pass for both probabilities. Keep the
checkpoint's MIT license and source attribution with the artifact.

### Breeze mouth

Breeze is a separate audio package using the same brain and endpoint. Supply the
Breeze source checkpoint, including its `audio_tokenizer/` directory:

```sh
python -m mtplx.frankie.prepare \
  --mouth-type breeze --gguf "$frankie_gguf" --ear "$ear_source" \
  --mouth '/absolute/path/to/breeze-source' \
  --output "$PWD/models/Frankie-Breeze-audio-q8"
```

Pass that new folder as `--audio` when starting the server. Keep it separate from
the Qwen3-TTS package. Breeze receives the generated spoken text through its
native text encoder; brain hidden states select its emotion instruction. No
learned hidden-state-to-text bridge supplies the words. Its depth decoder reuses
the existing MLX layers with an incremental KV cache. Reference WAV uploads and
launch-time voice overrides work with either mouth.

The preparer preserves cached Breeze voice conditioning when supplied by the
GGUF; otherwise it computes conditioning from the GGUF's reference in memory.
Eligible linear and embedding weights use 8-bit quantization, while the codec
retains source precision. Keep the Breeze checkpoint's license with the artifact.

Breeze reuses its acoustic backbone KV cache across completed speech chunks in
one response. A continuation appends the speech boundary and new text to the
evaluated history, using mlx-audio's existing generation and streaming codec.
It does not re-encode or replay previous speech. This needs no model conversion
and works with packaged and custom voices; Qwen3-TTS keeps its existing behavior.

`MTPLX_FRANKIE_SPEECH_CONTEXT_WORDS` defaults to `100`; set it to `0` to disable
reuse, or select up to `1000`. `MTPLX_FRANKIE_BREEZE_CONTEXT_ROWS` defaults to
`2048` and accepts `0` or `1024` through `8192`. Before the word or row capacity
would be exceeded, generation starts again from the voice reference, reserving
space for the new chunk's maximum audio length. This periodically refreshes the
whole history instead of shifting individual KV rows. Completed caches larger
than 100,000,000 bytes, including allocation padding, are discarded regardless
of those settings. That limit applies to retained KV arrays, not temporary
inference buffers or the MLX allocator's reusable pool.

Responses, voice changes, interruptions and incomplete generation clear the
cache. In a controlled 100-word-context measurement with all Frankie weights
resident, the longest retained history used 86.5 MB and increased overall peak
active memory by about 9.9 MB versus independent chunks. These are measurements
of that workload, not a bound on total server memory.

For a brain finetune, change `brain_source` and run Forge again with that
checkpoint's MTP head. The currently trained Frankie adapters expect the
compatible 27B architecture and layer-16 features. A different architecture or
hidden width requires new adapters and validation; changing a path alone does
not establish compatibility.

## Start the server and page

```sh
export MTPLX_FRANKIE_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(24))')"
export MTPLX_FRANKIE_IDLE_HISTORY_PREFILL=1
mtplx frankie \
  --brain '/absolute/path/to/Frankie-brain-q4-mtp' \
  --audio "$PWD/models/Frankie-audio-q8" \
  --mtp 3 --host 127.0.0.1 --port 18870
```

After the log reports `Frankie ready`, open
`http://127.0.0.1:18870/#YOUR_TOKEN`, replacing `YOUR_TOKEN` with the environment
variable value. Click **Start conversation**, allow microphone access, and speak.
The page also accepts typed messages and images. **Conversation settings** has
thinking levels and a custom WAV reference with an optional transcript. A voice
uploaded through the page lasts for that conversation; disconnect restores the
server's default. To select a server default at launch, add
`--voice /absolute/path/to/reference.wav --voice-transcript 'Words in the clip'`.
If no transcript is supplied, the in-process ear transcribes the reference.
References must contain 1–30 seconds of speech.

Thinking defaults to off. The API accepts `off`, `minimal`, `low`, `medium`,
`high`, `xhigh`, and `max` (also `reasoning.effort: none` for off). Their reasoning budgets are 0,
64, 256, 1024, 4096, 16384, and 32768 tokens. These levels set token caps; the
model's template retains its default reasoning policy. `max_output_tokens` limits the entire response;
raise it when using a larger thinking budget. The page adjusts this limit to
leave 512 answer tokens beyond the selected reasoning budget. Reasoning and tool syntax are
excluded from spoken output. Wait for the current response to finish before
changing settings.

Voice and HTTP share brain sampling defaults: thinking uses temperature 1.0,
top-p .95; non-thinking uses temperature .7, top-p .8. Both use top-k 20,
min-p 0 and repetition penalty 1. HTTP presence penalty defaults to 0 for thinking
and 1.5 for non-thinking; voice retains presence penalty 0. HTTP requests accept
`reasoning_effort` or `enable_thinking`, with explicit sampling overrides taking
precedence. `presence_penalty` and `frequency_penalty` count generated tokens
only. Nonzero `min_p` and repetition penalties other than 1 are currently
rejected instead of silently ignored. Temperature 0 remains available for
deterministic tests. Mouth sampling is independent of these brain settings.

To serve clients on your LAN, use `--host 0.0.0.0`. Clients use the Mac's LAN
address with the same port and token: `http://YOUR_MAC:18870/v1` for HTTP APIs
and `ws://YOUR_MAC:18870/v1/realtime` for Realtime. This host setting applies to
both APIs in the same process. For the browser demo's microphone, use HTTPS or
keep the server on loopback and tunnel it:

```sh
ssh -N -L 18870:127.0.0.1:18870 user@your-mac
```

Open the localhost page on the client machine. Browsers permit microphone access
on localhost or HTTPS. Treat the launch token as private. `/health` reports the
server PID and MTP depth. Only one active Realtime conversation is accepted;
HTTP completion requests can run alongside it.

### Prepare history between turns

The launch example enables `MTPLX_FRANKIE_IDLE_HISTORY_PREFILL=1`. After a client
confirms playback has finished, completed replies of at least 24 words can be
prepared in the existing session cache before the next user turn. This moves
history processing off the next turn's critical path without changing model
weights, sampling, or mouth settings.

This requires `frankie.playback.finished` and skips background-task mode and
pending input or HTTP work. New input, settings, or requests cancel unfinished
work at bounded prefill chunks. Tiny replies use the ordinary path without an
extra pass. Clients without playback feedback keep their existing behavior.

A paired test with an official Q8 27B brain, Breeze, and MTP 2 measured
median speech-end-to-first-received-audio latency of 785 to 615 ms at short context and 906 to 720 ms at 13k–14k tokens.
All six paired follow-up texts and PCM outputs matched exactly. These are small
controlled workloads, not latency guarantees or physical speaker measurements.

Preparation uses extra idle compute and can retain more cache memory: paired
active-memory differences ranged from about 25 MB to 1.4 GB in those tests.
The existing session cache limits still apply; no additional model is loaded.
The server default remains off. Set `MTPLX_FRANKIE_IDLE_HISTORY_PREFILL=0` or
omit the export to keep it off when memory or idle compute is constrained.

## Duplex and agent harnesses

`session.frankie.input_context` advertises optional context for the next user input.
Send `{"type":"frankie.input_context.update","revision":1,"text":"Current view information"}`
with an increasing unsigned revision and at most 16 KiB of UTF-8 text. The server
acknowledges `frankie.input_context.updated` without changing session instructions
or generating a reply. Speech onset latches the context (first append for manual
audio); it becomes an `input_text` part before the audio. Changes during speech
apply to the following turn. Typed user items share this context mechanism, and
clear/false-start recovery retain undelivered context. Transcription events use
the audio part's actual content index.

Connect a Realtime client to `ws://127.0.0.1:18870/v1/realtime` with
an `Authorization` bearer header using the `MTPLX_FRANKIE_TOKEN` value. The page uses the equivalent WebSocket
subprotocol authentication. Audio output is mono PCM16 at 24 kHz; input accepts
16 or 24 kHz, configured through `session.update`.

The socket continues receiving microphone frames during generation and playback.
CPU VAD detects resumed speech, discards uncommitted responses, and cancels active
speech. Clients clear their playback queue on `frankie.playback.clear` and report
`conversation.item.truncate` with the amount actually played. History retains only
completed speech chunks heard before that cutoff. The page uses a 120 ms playback
buffer and bounds queued speech to reduce interruptions and underruns.

When the package contains VAP weights, learned turn projection also considers
the microphone and actual speaker playback on the same sample clock. The page
sends its aligned playback reference alongside microphone frames. VAP runs on
the CPU and can hold a short pause open; stale predictions cannot stall a turn,
and 1.5 seconds of silence forces release. Older packages continue using VAD.

A tool-capable harness supplies function schemas in `session.update.tools` and
consumes function-call items in `response.done`. It executes tools through its
normal permission rules, inserts `function_call_output` items, and requests the
next response. The model server does not execute tools. For a harness that owns
turn scheduling, set `audio.input.turn_detection.create_response` to `false`;
committed microphone turns still arrive normally. The existing Hermes
`realtime-voice` plugin was exercised against this endpoint with its real agent
loop and terminal tool. Configure its endpoint, model `Frankie`, and access token;
its context and tools stay in Hermes while all model inference stays in MTPLX.

Supported events include session updates, conversation message and tool-history
insertion, audio append/commit/clear, response create/cancel, item retrieve/truncate,
and `frankie.voice.update`. Audio and transcript deltas use the
`response.output_audio.*` and `response.output_audio_transcript.*` event names.
This is the Frankie Realtime endpoint, not a claim that every Realtime API event
or MTPLX's other HTTP endpoints is implemented by this command.

## Verify your installation

Record a short mono PCM16 24 kHz WAV asking “What is two plus two?” Use your own
recording, not the packaged default voice. Keep the server running, close other
connected clients, and run in another terminal with the same token:

```sh
python scripts/frankie_smoke.py \
  --audio /absolute/path/to/arithmetic.wav \
  --output /absolute/path/to/frankie-results.json
```

The harness checks text and memory, generated image understanding, spoken output,
raw audio input, thinking, tool calls and results, input while speech is active,
cancellation with heard-prefix history, continued conversation, and a custom voice.
When speech is enabled, token timing includes speech generation and playback
pacing; it is not an isolated brain-throughput benchmark.
It saves measured response and first-audio times rather than asserting a hardware
independent latency target. Test natural barge-in and device playback in the page
as well; a protocol test does not measure speakers or microphone echo behavior.

```sh
python -m pytest tests/test_committed_features.py tests/test_frankie_session.py \
  tests/test_frankie_audio.py \
  tests/test_generation_sustained.py tests/test_no_mlx_imports.py \
  tests/test_public_cli.py tests/test_runtime_kpis.py
python -m build
scripts/fresh_venv_smoke.sh
```

The feature tests cover rejected MTP rows, pending committed tokens, and cleanup.
The session tests cover unpublished speculation, tool-history replay, duplicate
tool results, settings changes with pending PCM, VAD timing, and heard-prefix
truncation. The Frankie command deliberately uses eager `capture_commit`
verification: its executed-feature callback is not supported by compiled verify.
MTP remains enabled; `--mtp 0` selects autoregressive generation for comparisons.

For the browser playback and natural VAD barge-in test, install Playwright in an
ignored tools directory and run the included browser harness with the same WAV:

```sh
npm install --prefix outputs/browser-tools playwright@1.60.0
outputs/browser-tools/node_modules/.bin/playwright install chromium
export FRANKIE_PLAYWRIGHT_MODULE="$PWD/outputs/browser-tools/node_modules/playwright/index.mjs"
node scripts/frankie_browser.mjs --audio /absolute/path/to/arithmetic.wav
```

It drives the actual page and AudioWorklet, feeds the recording at microphone
speed, interrupts an active spoken response, checks a longer response and
reconnection, and fails on browser errors or playback underruns. Its JSON and
screenshots stay in `outputs/`. It measures the browser audio path using a
synthetic microphone source; still listen through your own devices to assess
hardware and acoustic quality.


## Recorded integration run

On 2026-09-11, the M3 Ultra 256 GiB run used the source revision above, a
4-bit affine brain (group 64), BF16 MTP head, 8-bit eligible audio weights,
MTP depth 3, and the sustained profile with eager verification. Fan control was
not changed. The combined native package occupied 19,883,160,968 bytes including
its initial manifest and hashes; this is disk size, not a RAM requirement.

The public protocol harness passed ten checks. Warm first-PCM latency was
0.40–0.64 seconds for its short speech cases. The browser harness passed repeated
live voice turns, VAD barge-in, a 32-second spoken response, and reconnecting,
with zero measured playback underruns or JavaScript errors. Short arithmetic
turns first delivered audio about 0.87–0.94 seconds after the recorded utterance
ended, including VAD and the connection to the server. Browser sampling used the
page defaults (temperature 0.7), while the protocol cases used temperature 0.
These are functional integration timings, not isolated decode benchmarks.

A separate test through Hermes' normal agent loop executed a real terminal
command, spoke its result, answered live PCM input, restored tool history, and
recalled the result after changing the thinking level. A real-model edge probe
also covered one-, two-, and four-token limits, Unicode text, invalid input
recovery, and exclusive conversation ownership.

## Experimental concurrent completions

The Frankie server also exposes `/v1/chat/completions`, `/v1/completions`, and
`/v1/models` using the same bearer token as Realtime. Standard OpenAI SDK clients
can select its base URL. The loaded brain and vision weights are shared with
voice. The selected `--mtp` depth applies to both voice and HTTP completions.
HTTP requests keep separate prompt and draft caches and advance one committed
MTP cycle at a time on the same model worker. With `--mtp 0`, HTTP uses mlx-lm
autoregressive batching.

`--http-slots` defaults to four (maximum eight), and `--http-ctx-size` defaults
to 4096 tokens per request, including its output budget. These limits apply to
HTTP requests, independently of the voice conversation. HTTP caches are created
on demand and released as requests finish. HTTP caches currently use the
runtime's ordinary cache precision; this path does not promise Q4 KV.

Chat content uses ordered `text` and `image_url` parts. Image URLs can be inline
base64 data URLs or HTTP(S) URLs. Downloads happen outside the model worker so
a slow image host cannot block speech. Each request accepts at most four images,
12 MiB per image, and an 18 MiB body. Both streamed SSE and non-streamed results
are supported, including `stream_options.include_usage`. `/v1/completions` is
the legacy text-prompt endpoint, not an image API.

Realtime images use `conversation.item.create` with a user message and
`{"type":"input_image","image_url":"data:image/png;base64,..."}` content.
Supply inline PNG or JPEG data. Both interfaces accept `detail` values `auto`,
`low`, and `high`; preprocessing uses the model's image budget rather than
OpenAI-specific resolution tiers. Image and text parts keep their input order.

During overlapping requests, up to 800 ms of speech can be buffered to protect
playback while text/image prompts advance. The voice-only path keeps its
existing buffer target. Requests have separate histories, cancellation, and
samplers. This remains an experimental API subset: unsupported parameters
return errors, and `/v1/responses` is not implemented.
