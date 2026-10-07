"""MaAI acoustic turn-taking: decide from sound whether an overlap is a nod.

While Frankie talks, the user's microphone and Frankie's own playback are fed
in 80 ms frames to MaAI's BC-Det head (streaming Mimi encoder, 20 s cache).
A hysteresis gate turns its backchannel probability into one decision: once
the user has been voiced for three frames and the probability stays low for
``confirmations`` frames, the overlap is real speech and Frankie yields.
Anything never yielded stays a backchannel; no post-utterance ASR or brain
classification runs. This replaces the session's own prefix listener.

The detector is one shared CPU model; each session owns a worker thread that
resets its recurrent state. Weights come from a directory holding
``bc_det.pt``, ``mimi.onnx`` and ``mimi.json``.
"""
import asyncio
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import queue
import sys
import threading
import time
import types

import numpy as np

FRAME = 1280  # 80 ms at 16 kHz
FEATURES = dict(interruption_policy="semantic", playback_pause=False,
                streaming_listener="backchannel", background_tasks=False)


def _import_bc_det():
    # MaAI's package __init__ pulls in microphone and GUI modules; load only
    # the inference modules, from wherever ``maai`` is installed.
    if "maai" not in sys.modules:
        spec = importlib.util.find_spec("maai")
        if spec is None:
            raise ImportError("MaAI turn-taking needs the maai package (maai==0.2.18)")
        package = types.ModuleType("maai")
        package.__path__ = list(spec.submodule_search_locations)
        sys.modules["maai"] = package
    from maai.models.bc_det import BcDetGPT
    from maai.models.config import VapConfig
    return BcDetGPT, VapConfig


class Detector:
    """Unmodified MaAI encoder and BC-Det head, upstream step and overlap."""

    def __init__(self, weights, threads=4):
        import torch
        BcDetGPT, VapConfig = _import_bc_det()
        weights = Path(weights)
        conf = VapConfig(frame_hz=12.5, encoder_type="mimi", context_limit=250)
        conf.runtime_device = "cpu"
        conf.mimi_onnx_fp32_path = str(weights / "mimi.onnx")
        conf.mimi_onnx_fp32_meta_path = str(weights / "mimi.json")
        conf.mimi_onnx_cpu_intra_threads = threads
        conf.mimi_onnx_cpu_inter_threads = 1
        self.model = BcDetGPT(conf)
        self.model.load_encoder("")
        state = torch.load(weights / "bc_det.pt", map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        # Both channel encoders share one frame-rate convolution, as Maai.__init__ does.
        for suffix in ("weight", "bias"):
            value = state.pop("encoder.frame_rate_conv." + suffix)
            for channel in (1, 2):
                state[f"encoder{channel}.frame_rate_conv.{suffix}"] = value
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"BC-Det weights do not fit: {missing} {unexpected}")
        self.model.eval()
        self.reset()

    def reset(self):
        self.cache = None
        self.previous = [np.zeros(320, np.float32), np.zeros(320, np.float32)]
        self.first = True
        self.model.encoder1.reset_streaming_state()
        self.model.encoder2.reset_streaming_state()

    def process(self, user, played):
        """Return (backchannel probabilities or None, compute ms) for one frame."""
        import torch
        assert user.shape == played.shape == (FRAME,)
        start = time.perf_counter()
        with torch.inference_mode():
            inputs = [np.concatenate((p, x)) for p, x in zip(self.previous, (user, played))]
            self.previous = [x[-320:].copy() for x in (user, played)]
            a, b = self.model.encode_audio(*(torch.from_numpy(x)[None, None] for x in inputs))
            if self.first:
                self.first = False
                return None, (time.perf_counter() - start) * 1000
            assert a.shape == b.shape == (1, 1, 256), (a.shape, b.shape)
            result, cache = self.model(a, b, cache=self.cache)
            self.cache = {k: ([t[..., -249:, :] for t in ks], [t[..., -249:, :] for t in vs])
                          for k, (ks, vs) in cache.items()}
        return result["p_bc_det"], (time.perf_counter() - start) * 1000


@dataclass
class Gate:
    """Causal backchannel gate; one score arrives per 80 ms frame."""

    threshold: float = .2
    release_threshold: float = .05
    confirmations: int = 2
    seen_bc: bool = False
    low_run: int = 0
    high_run: int = 0
    last_end: float = -1

    def observe(self, probability, audio_end, voiced):
        """Return True once the overlap is confirmed not to be a backchannel."""
        if audio_end <= self.last_end:
            return False
        self.last_end = audio_end
        if not voiced:
            self.low_run = 0
            self.high_run = 0
            return False
        self.high_run = self.high_run + 1 if probability >= self.threshold else 0
        if self.high_run >= self.confirmations:
            self.seen_bc = True
        threshold = self.release_threshold if self.seen_bc else self.threshold
        self.low_run = self.low_run + 1 if probability < threshold else 0
        return self.low_run >= self.confirmations


class Worker:
    """One session's detector thread, fed by the session's turn audio."""

    def __init__(self, session, detector):
        self.session = session
        self.detector = detector
        self.queue = queue.Queue(maxsize=32)
        self.stop = threading.Event()
        self.epoch = 0
        self.thread = threading.Thread(target=self.work, daemon=True)
        detector.reset()
        self.thread.start()

    def append(self, user, played, end_ms):
        if self.stop.is_set():
            return
        if self.queue.full():
            # A burst must not end the websocket. Drop stale classifier work
            # (never the session's microphone audio) and resync the detector.
            self.epoch += 1
            self.drain()
            self.session.maai_latest = None
            self.session.maai_gate_id = None
            self.session.event("bench.detector_resync", epoch=self.epoch, audio_end_ms=end_ms)
        self.queue.put_nowait((self.epoch, user.copy(), played.copy(), end_ms, time.monotonic()))

    def drain(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def work(self):
        user = np.empty(0, np.float32)
        played = np.empty(0, np.float32)
        epoch = self.epoch
        try:
            while not self.stop.is_set():
                item = self.queue.get()
                if item is None:
                    return
                current_epoch, a, b, end, queued = item
                if current_epoch != epoch:
                    self.detector.reset()
                    user = np.empty(0, np.float32)
                    played = np.empty(0, np.float32)
                    epoch = current_epoch
                user = np.concatenate((user, a))
                played = np.concatenate((played, b))
                while len(user) >= FRAME and not self.stop.is_set():
                    frame_end = end - (len(user) - FRAME) / 16
                    probabilities, compute = self.detector.process(user[:FRAME], played[:FRAME])
                    user, played = user[FRAME:], played[FRAME:]
                    if probabilities is not None:
                        record = dict(epoch=epoch, audio_end_ms=frame_end, p=probabilities,
                                      compute_ms=compute,
                                      queue_to_result_ms=(time.monotonic() - queued) * 1000)
                        self.session.loop.call_soon_threadsafe(_result, self.session, record)
        except Exception as exc:
            error = str(exc)
            self.session.loop.call_soon_threadsafe(
                lambda: self.session.event("bench.detector_error", error=error))

    def close(self):
        self.stop.set()
        self.drain()
        self.queue.put_nowait(None)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("MaAI worker did not close")


def _result(session, result):
    worker = session.maai_worker
    if (session.closed or worker is None or worker.stop.is_set()
            or result["epoch"] != worker.epoch):
        return
    session.maai_latest = result
    session.event("bench.bc_probability", **result, clock_ms=session.clock_ms)
    _maybe_yield(session)


def _maybe_yield(session):
    result = getattr(session, "maai_latest", None)
    if (not result or not session.listening
            or result["audio_end_ms"] < session.speech_start_ms
            or session.clock_ms - result["audio_end_ms"] > 240):
        return
    if getattr(session, "maai_gate_id", None) != session.speech_id:
        session.maai_gate_id = session.speech_id
        session.maai_gate = Gate(*session.maai_gate_args)
    gate = session.maai_gate
    fresh = result["audio_end_ms"] > gate.last_end
    decision = gate.observe(result["p"][0], result["audio_end_ms"], session.voice_run > 0)
    if fresh:
        session.event("bench.gate", p=result["p"][0], audio_end_ms=result["audio_end_ms"],
                      voice_run=session.voice_run, speech_start_ms=session.speech_start_ms,
                      clock_ms=session.clock_ms, low_run=gate.low_run, seen_bc=gate.seen_bc,
                      decision=decision, qualified=session.voice_run >= 3)
    if session.voice_run >= 3 and decision:
        session.yield_prefix(session.overlap_run, "bench_acoustic_confirmed_non_backchannel")


class _TurnTee:
    def __init__(self, session, turn):
        self.session, self.turn = session, turn

    def __getattr__(self, name):
        return getattr(self.turn, name)

    def append(self, a, b):
        self.turn.append(a, b)
        self.session.maai_worker.append(a, b, self.session.clock_ms)


def install(detector, *, threshold=.2, release_threshold=.05, confirmations=2):
    """Make every session take turns by MaAI, with the demo's settings forced."""
    from mtplx.frankie.session import Session

    original_init, original_handle = Session.__init__, Session.handle
    original_info, original_close = Session.info, Session.close
    gate_args = (threshold, release_threshold, confirmations)

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.maai_worker = None
        self.maai_gate_args = gate_args

    async def handle(self, event):
        if self.maai_worker is None:
            self.maai_worker = await asyncio.to_thread(Worker, self, detector)
            self.turn = _TurnTee(self, self.turn)
        if event["type"] == "session.update":
            settings = dict(event.get("session", {}))
            settings["frankie"] = {**settings.get("frankie", {}), **FEATURES}
            event = {**event, "session": settings}
        return await original_handle(self, event)

    def append_prefix(self, frame):
        _maybe_yield(self)

    def schedule_overlap(self, item, run):
        # An overlap that was never yielded stays a backchannel.
        self.event("bench.backchannel_preserved", item_id=item["id"])

    def info(self):
        result = original_info(self)
        result["frankie"]["backchannel_detector"] = {
            "name": "maai-bc-det", "preset": "candidate-02",
            "threshold": threshold, "release_threshold": release_threshold,
            "enter_confirmations": confirmations, "exit_confirmations": confirmations,
        }
        return result

    async def close(self):
        if self.maai_worker:
            await asyncio.to_thread(self.maai_worker.close)
            self.maai_worker = None
        await original_close(self)

    Session.__init__, Session.handle, Session.info, Session.close = __init__, handle, info, close
    Session.append_prefix, Session.schedule_overlap = append_prefix, schedule_overlap
