"""Background task protocol tests need no model weights or MLX execution."""

import copy
import json

import pytest

from mtplx.frankie.tasks import TaskLedger, project_history


def call(call_id="call_1"):
    return {
        "id": "item_" + call_id,
        "type": "function_call",
        "call_id": call_id,
        "name": "lookup",
        "arguments": '{"city":"Paris"}',
    }


def result(call_id="call_1"):
    return {
        "id": "result_" + call_id,
        "type": "function_call_output",
        "call_id": call_id,
        "output": '{"temperature":18}',
    }


def test_task_ledger_accepts_completion_once_and_never_revives_cancelled_work():
    ledger = TaskLedger()
    ledger.register(call(), 3, "response_1")
    task, changed = ledger.cancel("call_1", "superseded")
    assert changed and task.status == "superseded"
    assert not ledger.cancel("call_1")[1]
    task, accepted = ledger.complete("call_1")
    assert not accepted and task.status == "superseded"
    with pytest.raises(ValueError, match="already supplied"):
        ledger.complete("call_1")
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.register(call(), 4)


def test_task_ledger_bounds_pending_work_and_terminal_tombstones():
    ledger = TaskLedger(max_pending=2, max_entries=3)
    for name in ("a", "b"):
        ledger.register(call(name), 0)
    with pytest.raises(ValueError, match="Too many pending"):
        ledger.register(call("c"), 0)
    ledger.complete("a")
    ledger.register(call("c"), 0)
    ledger.complete("c")
    with pytest.raises(ValueError, match="undelivered"):
        ledger.register(call("d"), 0)
    ledger.tasks["a"].consumed = True
    ledger.register(call("d"), 0)
    assert set(ledger.tasks) == {"b", "c", "d"}
    assert ledger.tasks["b"].status == "running"


def test_projection_preserves_prefix_and_chronological_knowledge_without_mutation():
    user = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Tell me a joke while waiting."}],
    }
    history = [call(), user]
    original = copy.deepcopy(history)
    pending = project_history(history)
    assert pending[0] == call()
    assert pending[1]["type"] == "function_call_output"
    assert json.loads(pending[1]["output"])["status"] == "pending"
    history.append(result())
    completed = project_history(history)
    assert completed[: len(pending)] == pending
    assert completed[-1]["role"] == "user"
    assert "Background task notice" in completed[-1]["content"][0]["text"]
    assert "18" in completed[-1]["content"][0]["text"]
    assert history[:-1] == original
    assert history[-1] == result()
    # Late cancelled results never enter the model-facing view.
    history[-1]["_task_discarded"] = True
    assert project_history(history) == pending
