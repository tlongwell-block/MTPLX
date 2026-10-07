import mlx.core as mx

from mtplx.frankie.demo.__main__ import seed_sessions


def test_each_session_is_seeded_once_at_its_first_reply():
    class Engine:
        def respond(self, items, settings, emit, abort, *, session_id):
            return mx.random.uniform().item()

    seed_sessions(Engine, seed=7)
    engine = Engine()
    first = engine.respond([], {}, None, None, session_id="a")
    later = engine.respond([], {}, None, None, session_id="a")
    other = engine.respond([], {}, None, None, session_id="b")
    mx.random.seed(7)
    assert first == other == mx.random.uniform().item()
    assert later != first
