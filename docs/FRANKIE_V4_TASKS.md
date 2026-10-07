# Experimental background conversations and playback feedback

This branch adds opt-in conversation scheduling without changing model weights or
executing tools inside the server. The connected harness still executes tools and
owns their permissions, cancellation, and side effects.

Enable the experiment in a standard Realtime session update:

```json
{
  "type": "session.update",
  "session": { "frankie": { "background_tasks": true } }
}
```

With the option omitted or false, existing response scheduling remains in place.
The server reports the active setting in `session.frankie.background_tasks`.

## Tool lifecycle

Calls and results use the usual `function_call`, `function_call_output`, and
`response.create` events. Users can speak or type while a call is pending. A result
can arrive while Frankie speaks; it does not mutate the running response's prompt
snapshot. A following `response.create` is coalesced and deferred until that
response is finished, its reported playback has drained, and no user utterance is in progress. New user input takes
priority and incorporates available results in its own next response.

The opt-in mode ignores duplicate/stale `response.create` events when there is no
new committed user input or unconsumed result. To request continuation, send an
explicit user message. It does not produce unsolicited result speech without a
client `response.create`.

A task update has this shape:

```json
{
  "type": "frankie.task.updated",
  "task": {
    "call_id": "call_example",
    "name": "lookup",
    "revision": 3,
    "response_id": "resp_example",
    "status": "running"
  }
}
```

Statuses are `running`, `completed`, `cancelled`, and `superseded`. Here `running`
means the call has been requested and no completion has been reported; the server
does not observe the external tool process. Imported calls can have a null response
ID. Up to 32 tasks can be pending; the ledger retains 256 entries including recent
terminal states. Canonical conversation history independently rejects duplicate
call IDs and outputs.

To suppress a pending result and ask the harness to stop its work:

```json
{"type":"frankie.task.cancel","call_id":"call_example","reason":"cancelled"}
```

Use `reason:"superseded"` when a request has been replaced. The resulting update
contains `cancellation_requested:true` for pending work. This is a request to the
harness, **not confirmation that an operation stopped or was undone**. The harness
must cancel its task if possible. Late results remain acknowledged as conversation
items, but carry `result_discarded:true` on the task update and are excluded from
future model prompts. Duplicate results are rejected.

Superseding an already completed task suppresses its retained result and any
queued result-only response, with `cancellation_requested:false`; it cannot unsay
content already spoken or retract information from an active prompt snapshot.
No task is implicitly cancelled merely because another user turn arrives.

## Chronological prompt projection

Wire conversation items are retained. Internally, every function call is paired
with a truthful pending observation so a new user turn does not leave an unmatched
call in the chat template. The final result is provided at its actual arrival
position as an explicitly labeled background-task data notice. It is not inserted
back beside the original call. This preserves the pending prompt prefix and avoids
pretending the result was known during intervening conversation.

The system guidance tells the brain that these notices contain tool data, not user
requests or instructions, and that pending work must not be issued again. This
establishes a mechanism, not proof that a particular model will always make the
correct conversational choice. Live tool-use evaluation is still required.

## Playback feedback

`session.frankie.playback_feedback:true` advertises optional browser feedback:

```json
{"type":"frankie.playback.position","item_id":"item_example","response_id":"resp_example","audio_end_ms":480}
{"type":"frankie.playback.finished","item_id":"item_example","response_id":"resp_example","audio_end_ms":1200}
```

Send integer positions monotonically, no more often than four times per second.
Positions cannot exceed the emitted PCM duration. `finished` means the response is
complete **and its audio actually drained**, not merely that `response.done` was
received. A one-millisecond rounding tolerance is permitted. Sixteen recent
responses are retained for delayed feedback; unknown/stale IDs are rejected.

A completed/drained response cannot subsequently acquire a false interruption
marker from an equally complete truncate event. Partial interruption still commits
only fully played speech chunks; word-aligned history is future work.

## Experimental semantic interruptions

The direct webpage offers Baseline and v4 experimental modes. A harness can enable
the same behavior with:

```json
{"type":"session.update","session":{"frankie":{"interruption_policy":"semantic","background_tasks":true,"playback_feedback":true}}}
```

The default policy is `vad`. `observe` retains VAD behavior and emits diagnostic
decisions without applying them. `semantic` uses the existing ear and a bounded
context on the already loaded brain to classify a completed overlapping utterance:

- `continue`: encouragement; keep speaking.
- `adapt`: a correction; discard unheard output and replan.
- `yield`: a new request; stop the old reply and respond to the new input.
- `stop`: an explicit request for silence; stop without another reply.
- `wait`: insufficient evidence; leave output unchanged.

This is final-utterance classification, not continuous semantic prefix listening.
The short probe shares the inference owner with speech generation and adds compute
and bounded cache memory. Its static instruction cache is separate from ordinary
voice/HTTP history. No second set of model weights or inference process is loaded.

For an accepted correction/new request, replanning receives the recognized words
as supplemental user text alongside the original neural audio features. The
transcript is labeled uncertain; incorrect recognition remains a quality risk.
Ordinary turns, stale decisions, and observation-only decisions do not receive
this supplement. Background speech and addressee recognition remain experimental.

Unresolved adjacent fragments from the same audible response are considered
together when their speech gap is at most 1.5 seconds. This prevents a short pause
from dropping the first half of a correction. Completed ear work is reused; each
original audio item and caption is retained. Accepted decisions consume the
fragments, and later turns or unrelated responses cannot reuse them.

Overlapping input exceeding six seconds across that group, including padding, returns to ordinary
turn handling and stops the old reply. The full recording is retained; the limit
does not truncate what the user said. Explicit cancellation remains immediate.
After `stop` or explicit cancellation, background results stay silent until fresh
user input; running tools are not implicitly cancelled.

The server emits `frankie.interaction.observed` with the action, application status,
and timing metrics. Action-only probes report `confidence:null`, not a fabricated
probability. `frankie.interaction` carries display state and any bounded fallback.
`assistant_name` can optionally identify the assistant for the classifier; its
default is `Frankie`.

## Independent validation

`tests/test_frankie_tasks.py` exercises bounded lifecycle state, duplicate
completion, cancellation, and immutable chronological prompt projection without
model weights. `tests/test_frankie_background_session.py` uses an explicitly gated
fake engine to cover pending tools plus new typed/spoken input, results arriving
mid-response, coalesced followups, stale triggers, superseded late results, and
playback completion versus drain. These tests do not establish naturalness,
latency, or production tool-harness compatibility; those require the live demo.

## Native reasoning boundaries

The voice path flushes completed public speech at a native `<think>` opener, so
an acknowledgment need not wait for the following private block to finish.
Private tokens do not enter TTS. Unfinished private blocks also stay out of HTTP
public content. Final tool parsing considers separate public spans only; it does
not dispatch tool-shaped text inside private reasoning or join incomplete calls
across a thinking boundary.

These rules do not cause an untrained model to interleave reasoning. They use the
request's existing thinking setting and preserve sampling and thinking budgets.
Conversation history still does not preserve arbitrary private/public segment
ordering, and tools dispatch at generation completion. A trained interleaving
model needs separate quality, history, and audio qualification.
