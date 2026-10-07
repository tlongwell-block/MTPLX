import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np
import pytest

from mtplx.frankie.breeze import BreezeModel
from mtplx.frankie.demo import delivery
from mtplx.frankie.demo.delivery import LEARNED, PLAIN, Delivery, Hold, Reader, calibrated, readings

WIDTH = 6


def directions(rng, axes):
    return dict(mu=rng.normal(size=WIDTH), V=rng.normal(size=(len(axes), WIDTH)),
                center=rng.normal(size=len(axes)), spread=1 + rng.random(len(axes)), axes=np.array(axes))


def calibration(k):
    return dict(ref=np.array(14), a=np.full(k, .5), b=np.full(k, 2.0), c=np.ones(k), e=np.full(k, 1.5))


@pytest.fixture
def assets(tmp_path):
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "directions.npz", **directions(rng, ["sad", "happy"]))
    np.savez(tmp_path / "words.npz", **directions(rng, ["calm", "brisk", "dry"]))
    np.savez(tmp_path / "calib-feelings.npz", **calibration(2))
    np.savez(tmp_path / "calib-words.npz", **calibration(3))
    bank = ["sad", "happy", "w-calm"]
    np.savez(tmp_path / "adapter.npz", bank=np.array(bank), mu=np.zeros(5), sd=np.ones(5),
             W=rng.normal(size=(6, 3)), A=np.r_[np.zeros(5), 3.3])
    (tmp_path / "rows").mkdir()
    for i, name in enumerate(bank):
        mx.save_safetensors(str(tmp_path / "rows" / f"{name}.slot.safetensors"),
                            {"theta": mx.full((1, 2, 4), float(i + 1))})
    return tmp_path


def test_calibration_leaves_reference_length_and_longer_unchanged():
    rng = np.random.default_rng(1)
    d, cal, mean = directions(rng, ["a", "b"]), calibration(2), rng.normal(size=WIDTH)
    for n in (14, 15, 200):
        np.testing.assert_allclose(readings(calibrated(d, cal, n), mean), readings(d, mean))
    # m(n) and s(n) grow on short phrases: the same state reads closer to ordinary.
    short = readings(calibrated(d, cal, 2), mean)
    assert not np.allclose(short, readings(d, mean))


def test_reader_weights_rows_and_strength(assets):
    reader = Reader(assets)
    assert reader.width == WIDTH and reader.bank == ["sad", "happy", "w-calm"]
    read = reader.read(np.random.default_rng(2).normal(size=WIDTH), 5)
    weights = read["weights"]
    assert abs(sum(weights.values()) - 1) < 2e-3 and all(x >= .05 for x in weights.values())
    assert list(weights.values()) == sorted(weights.values(), reverse=True)
    expected = sum(x * float(reader.bank.index(name) + 1) for name, x in weights.items())
    np.testing.assert_allclose(np.asarray(read["rows"]), expected, rtol=1e-6)
    assert read["strength"] == 3.5  # 3.3 on the half-step grid, inside 3..4
    assert read["feelings"].shape == (2,)


def test_reader_maps_another_brain_onto_the_adapter_scale(assets):
    mean = np.random.default_rng(3).normal(size=WIDTH)
    plain = Reader(assets).read(mean, 20)["feelings"]
    np.savez(assets / "flash-scale.npz", fa=np.full(2, 1.0), fc=np.zeros(2), fe=np.full(2, 2.0), fb=np.ones(2),
             wa=np.zeros(3), wc=np.zeros(3), we=np.ones(3), wb=np.ones(3))
    scaled = Reader(assets).read(mean, 20)["feelings"]
    np.testing.assert_allclose(scaled, np.round(1 + plain / 2, 2), atol=.011)


def test_learned_prompt_puts_rows_where_the_instruction_goes():
    ids = {"[S0]<ins_bos>x<ins_eos>": [7, 1, 9, 2], "[S0]hello": [7, 5, 6]}
    fake = NS(_speaker=lambda voice: "[S0]", tokenizer=NS(convert_tokens_to_ids=lambda token: 1),
              _text_ids=lambda text: mx.array(ids[text]),
              text_encoder=lambda x: x.astype(mx.float32)[..., None] * mx.ones((1, 1, 2)),
              text_encoder_proj=lambda x: x, instruction_rows=mx.full((1, 2, 2), -1.0))
    prompt = BreezeModel._learned_prompt(fake, "hello", None)
    assert prompt[0, :, 0].tolist() == [7, -1, -1, 5, 6]


class Mouth:
    def __init__(self):
        self.model = NS(generate=lambda *a, **k: iter(()), instruction_rows=None, hold_words=0)
        self.calls = []
        self.speak = self._speak

    def _speak(self, text, states, **kwargs):
        model = self.model
        self.calls.append((kwargs.get("speech_instruction"), model.instruction_rows is not None,
                           self.guidance.scale))
        yield np.zeros(2400, np.float32)


def led_mouth(assets, tmp_path, step=True):
    mouth = Mouth()
    reader = Reader(assets)
    d = Delivery(mouth, reader, log=tmp_path / "log")
    mouth.guidance = NS(scale=1.0, in_step=lambda: step)
    d._make_guidance = lambda: mouth.guidance
    return mouth, reader


def test_brain_leads_each_phrase_with_its_rows_and_strength(assets, tmp_path):
    mouth, reader = led_mouth(assets, tmp_path)
    states = mx.array(np.random.default_rng(4).normal(size=(5, WIDTH)), dtype=mx.bfloat16)
    pcm = list(mouth.speak("Hello there.", states, temperature=.9))
    expected = reader.read(np.asarray(mx.mean(states.astype(mx.float32), axis=0)), 5)
    assert len(pcm) == 1 and mouth.calls == [(LEARNED, True, expected["strength"])]
    assert mouth.guidance.scale == 1.0 and mouth.model.instruction_rows is None
    entry = json.loads((tmp_path / "log/delivery.jsonl").read_text())
    assert entry["weights"] == expected["weights"] and entry["led"] and entry["rows"] == 5
    assert np.load(tmp_path / "log/states/0.npy").shape == (5, WIDTH)


def test_plain_when_lanes_differ_states_are_missing_or_client_chose(assets, tmp_path):
    states = mx.zeros((3, WIDTH))
    mouth, _ = led_mouth(assets, tmp_path, step=False)
    list(mouth.speak("Hi.", states))
    assert mouth.calls == [(PLAIN, False, 1.0)]
    mouth, _ = led_mouth(assets, tmp_path / "b")
    list(mouth.speak("Hi.", mx.zeros((3, WIDTH + 1))))
    list(mouth.speak("Hi.", states, speech_instruction="Speak softly."))
    assert mouth.calls == [(PLAIN, False, 1.001), ("Speak softly.", False, 1.0)]


def test_install_wraps_engine_construction(assets, monkeypatch):
    class Engine:
        def __init__(self):
            self.audio = NS(breeze=Mouth())

    delivery.install(Engine, assets)
    engine = Engine()
    assert engine._expression_feature_stream.keywords == {"width": WIDTH}
    assert engine.audio.breeze.speak == engine._delivery.speak


def test_hold_drops_held_speech_only_for_an_extreme_feeling():
    hold, sad, happy, calm = Hold(), np.array([8.0, 0.0]), np.array([0.0, 6.0]), np.array([1.0, 1.0])
    assert hold.why(sad, True) is None  # nothing held yet
    hold.add(10, calm)
    assert hold.why(np.array([4.0, 0.0]), False) is None  # a feeling, not an extreme one
    assert hold.why(sad, False) == ("enter", 0, 8.0, 1.0)
    hold.clear(); hold.add(10, sad)
    assert hold.why(sad, False) is None  # the held voice already reads it
    assert hold.why(calm, False) is None  # leaving is only judged at a reply's start
    assert hold.why(calm, True) == ("leave", 0, 8.0, 1.0)
    assert hold.why(np.array([3.0, 0.0]), True) is None
    hold.add(5, happy)
    hold.follow(6)  # Breeze evicted the oldest phrase
    assert [w for w, _ in hold.held] == [5]


def test_override_resets_both_lanes_before_the_phrase(assets, tmp_path):
    mouth = Mouth()
    resets = []
    mouth.model.__dict__.update(hold_words=40, hold_speech=lambda: None, _speech_cache=[1], _context_words=12,
                                reset_speech_context=lambda: resets.append(True))
    reader = Reader(assets)
    d = Delivery(mouth, reader, log=tmp_path)
    mouth.guidance = NS(scale=1.0, in_step=lambda: True)
    d._make_guidance = lambda: mouth.guidance
    d.hold.add(12, np.array([0.0, 0.0]))
    reader.read = lambda mean, n: dict(weights={"sad": 1.0}, rows=mx.zeros((1, 1, 4)), strength=4.0,
                                       feelings=np.array([9.0, 0.0]))
    list(mouth.speak("So sorry.", mx.zeros((2, WIDTH))))
    assert resets == [True] and [w for w, _ in d.hold.held] == [2]
    entry = json.loads((tmp_path / "delivery.jsonl").read_text())
    assert entry["held"] == dict(rule="enter", feeling="sad", reading=9.0, held=0.0)
    mouth.model.hold_speech()  # the engine's per-reply call marks the next phrase as a reply's first
    assert d.reply_start


REPLAY = os.environ.get("MTPLX_FRANKIE_DELIVERY_REPLAY")


@pytest.mark.skipif(not REPLAY, reason="set MTPLX_FRANKIE_DELIVERY_REPLAY=ASSETS::RUN to replay saved live phrases")
def test_replay_saved_live_phrases_reproduces_every_decision():
    """RUN holds states/N.npy and a log with each phrase's weights, strength and feeling readings."""
    folder, run = (Path(x) for x in REPLAY.split("::"))
    reader = Reader(folder)
    log = run / ("delivery.jsonl" if (run / "delivery.jsonl").exists() else "brain-led.jsonl")
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert entries
    for n, entry in enumerate(entries):
        states = mx.array(np.load(run / f"states/{n}.npy"))
        read = reader.read(np.asarray(mx.mean(states.astype(mx.float32), axis=0)), len(states))
        assert read["weights"] == entry.get("weights", entry.get("v9")), n
        assert read["strength"] == entry["strength"], n
        assert read["feelings"].tolist() == entry.get("feelings", entry.get("zf")), n
