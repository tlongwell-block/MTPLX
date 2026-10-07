from types import SimpleNamespace as NS

from mtplx.frankie.demo.telemetry import _publish


def test_latency_stages_are_published_on_the_session_loop():
    events, scheduled = [], []
    session = NS(closed=False, event=lambda name, **v: events.append((name, v)),
                 loop=NS(call_soon_threadsafe=scheduled.append))
    record = dict(queued=1.0, owner=1.01, decode=1.5, token=1.6, tts=2.0, audio=2.25,
                  response_id="r", input_item_id="i", session=lambda: session)
    _publish(record)
    scheduled[0]()
    name, value = events[0]
    assert name == "frankie.latency" and value["response_id"] == "r"
    assert value["stages"] == {"queue_ms": 10.0, "input_prefill_ms": 490.0,
                               "first_token_ms": 100.0, "first_phrase_ms": 400.0,
                               "tts_first_chunk_ms": 250.0}


def test_missing_stages_and_closed_sessions_publish_nothing_wrong():
    events, scheduled = [], []
    session = NS(closed=True, event=lambda *a, **v: events.append(a),
                 loop=NS(call_soon_threadsafe=scheduled.append))
    _publish(dict(tts=1.0, audio=1.5, response_id="r", input_item_id=None,
                  session=lambda: session))
    scheduled[0]()
    assert events == []
    _publish(dict(response_id="r", input_item_id=None, session=lambda: None))
    assert len(scheduled) == 1
