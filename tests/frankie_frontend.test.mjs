import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const root = new URL("../mtplx/frankie/", import.meta.url);
const worklet = readFileSync(new URL("audio-worklet.mjs", root), "utf8");
const page = readFileSync(new URL("page.html", root), "utf8")
  .match(/<script type="module">([\s\S]*?)<\/script>/)[1];

function player() {
  let Processor;
  const events = [];
  vm.runInNewContext(worklet, {
    AudioWorkletProcessor: class {
      constructor() { this.port = { postMessage: (event) => events.push(event) }; }
    },
    registerProcessor: (_, value) => { Processor = value; },
    sampleRate: 24000,
  });
  const instance = new Processor();
  const send = (data) => instance.port.onmessage({ data });
  const render = (count = 128) => {
    const output = new Float32Array(count);
    instance.process([[new Float32Array(count)]], [[output]]);
    return output;
  };
  const add = (count = 3072, itemId = "item") => send({
    type: "audio", responseId: "reply", itemId,
    pcm: new Int16Array(count).fill(8192),
  });
  const done = () => send({ type: "done", itemId: "item" });
  const clear = () => {
    send({ type: "clear", requestId: "next-utterance" });
    return events.at(-1);
  };
  return { events, send, render, add, done, clear };
}

function browser() {
  const elements = new Map(), sent = [], audio = [];
  const timers = new Map();
  let timerId = 0;
  let now = 0;
  const element = () => ({
    textContent: "", children: [], value: "", checked: false,
    style: { setProperty() {} }, scrollTop: 0, scrollHeight: 100,
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren() { this.children = []; },
  });
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  get("thinking").value = "off";
  get("speak").checked = true;
  get("mode").value = "v5";
  const context = vm.createContext({
    document: { getElementById: get, createElement: element }, window: {},
    performance: { now: () => now },
    atob: (value) => Buffer.from(value, "base64").toString("binary"),
    setTimeout: (callback) => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: (id) => timers.delete(id),
  });
  vm.runInContext(page + `
    globalThis.api = {
      onEvent, onPlaybackState, settings, feedback, end,
      evidence,
      attach(socket, processor) { ws = socket; node = processor; },
      get active() { return active; },
    };
  `, context);
  context.api.attach(
    { readyState: 1, send: (data) => sent.push(JSON.parse(data)), close() {} },
    { port: { postMessage: (data) => audio.push(data) } },
  );
  return {
    api: context.api, get, sent, audio, clock: (value) => { now = value; },
    finishTimers() {
      const callbacks = [...timers.values()]; timers.clear();
      for (const callback of callbacks) callback();
    },
  };
}

test("demo keeps acknowledgments uninterrupted while reporting heard position", () => {
  const b = browser(); b.api.settings();
  assert.equal(b.sent.at(-1).session.frankie.playback_pause, false);
  assert.equal(b.sent.at(-1).session.frankie.playback_feedback, true);
});

test("completed transcripts replace placeholders and generated text beyond heard cutoff", async () => {
  const b = browser();
  await b.api.onEvent({ type: "conversation.item.created", item: {
    id: "user", type: "message", role: "user", content: [{type: "input_audio"}],
  }});
  await b.api.onEvent({type: "conversation.item.input_audio_transcription.completed", item_id: "user", transcript: "Please continue."});
  assert.equal(b.get("log").children[0].children[1].textContent, "Please continue.");
  await b.api.onEvent({type: "response.output_audio_transcript.delta", item_id: "answer", delta: "Heard. Not yet heard."});
  await b.api.onEvent({type: "response.output_audio_transcript.done", item_id: "answer", transcript: "Heard."});
  assert.equal(b.get("log").children[1].children[1].textContent, "Heard.");
});

test("local diagnostics retain stop and listener evidence without raw media or arbitrary metrics", async () => {
  const b = browser();
  await b.api.onEvent({type: "frankie.metrics", response_id: "reply", metrics: {
    generated_tokens: 2, finish_stop_origin: "initial_target", audio_seconds: 0.5,
    cached_tokens: 2048, arbitrary_private_payload: "do not retain",
  }});
  await b.api.onEvent({type: "frankie.interaction.prefix", action: "wait", gate: "newer_audio",
    transcript: "No, keep going", apply: false, audio: "do not retain"});
  await b.api.onEvent({type: "response.done", response: {id: "reply", status: "completed"}});
  const events = JSON.parse(JSON.stringify(b.api.evidence));
  assert.equal(events[0].metrics.finish_stop_origin, "initial_target");
  assert.equal(events[0].metrics.generated_tokens, 2);
  assert.equal(events[1].transcript, "No, keep going");
  assert.equal(events[1].apply, false);
  assert.equal(events[2].status, "completed");
  assert.equal(events[2].response_id, "reply");
  assert.ok(!JSON.stringify(events).includes("do not retain"));
});

test("pause and resume reach queued audio even after generation finishes", async () => {
  const b = browser();
  await b.api.onEvent({ type: "response.created", response: { id: "reply" } });
  await b.api.onEvent({ type: "response.done", response: { id: "reply", status: "completed" } });
  assert.equal(b.api.active, false);
  await b.api.onEvent({ type: "frankie.playback.pause", response_id: "reply" });
  await b.api.onEvent({ type: "frankie.playback.resume", response_id: "reply" });
  assert.deepEqual(JSON.parse(JSON.stringify(b.audio)), [
    { type: "pause", responseId: "reply" }, { type: "resume", responseId: "reply" },
  ]);
  assert.equal(b.sent.filter((e) => e.type === "conversation.item.truncate").length, 0);
});

test("completed playback retires after drain, before a new user turn", () => {
  const p = player(); p.add(); p.done();
  const output = p.render(3200);
  assert.ok(output.subarray(0, 3072).every((x) => x === 0.25));
  assert.ok(output.subarray(3072).every((x) => x === 0));
  const finished = p.events.filter((e) => e.type === "finished");
  assert.equal(finished.length, 1);
  assert.equal(finished[0].playback.playedSamples, 3072);
  assert.equal(p.clear().playback, null);
  p.render(); p.done();
  assert.equal(p.events.filter((e) => e.type === "finished").length, 1);
});

test("completion after final render also retires the item", () => {
  const p = player(); p.add(); p.render(3072); p.done();
  assert.equal(p.clear().playback, null);
});

test("interruption of completed but still queued speech reports exact playback", () => {
  const p = player(); p.add(); p.done(); p.render(128);
  assert.equal(p.clear().playback.playedSamples, 128);
  p.add();
  assert.ok(p.render().every((x) => x === 0));
  assert.equal(p.events.filter((e) => e.type === "finished").length, 0);
});

test("temporary underrun remains interruptible until generation completes", () => {
  const p = player(); p.add(); p.render(3200);
  assert.equal(p.events.filter((e) => e.type === "finished").length, 0);
  assert.equal(p.clear().playback.playedSamples, 3072);
});

test("short complete speech drains without waiting for the startup buffer", () => {
  const p = player(); p.add(48); p.done();
  assert.ok(p.render(48).every((x) => x === 0.25));
  assert.equal(p.clear().playback, null);
});

test("late completion for an older item cannot retire a new item", () => {
  const p = player(); p.add(); p.done(); p.render(3072);
  p.add(3072, "next"); p.done(); p.render(128);
  assert.equal(p.clear().playback.itemId, "next");
});

for (const policy of [undefined, "semantic"]) {
  test(`only authoritative server clear interrupts playback (${policy || "baseline"})`, async () => {
    const b = browser();
    await b.api.onEvent({ type: "session.updated", session: { frankie: { interruption_policy: policy } } });
    await b.api.onEvent({ type: "input_audio_buffer.speech_started" });
    assert.equal(b.audio.filter((e) => e.type === "clear").length, 0);
    await b.api.onEvent({ type: "frankie.interaction", state: "continue" });
    assert.equal(b.audio.filter((e) => e.type === "clear").length, 0);
    await b.api.onEvent({ type: "frankie.playback.clear" });
    assert.equal(b.audio.filter((e) => e.type === "clear").length, 1);
  });
}

test("baseline sessions do not opt into experimental controls", () => {
  const b = browser();
  b.get("mode").value = "baseline"; b.api.settings();
  assert.equal(b.sent.at(-1).session.frankie, undefined);
  b.get("mode").value = "v5"; b.api.settings();
  assert.equal(b.sent.at(-1).session.frankie.interruption_policy, "semantic");
  assert.equal(b.sent.at(-1).session.frankie.background_tasks, true);
});

test("playback feedback is capability gated and throttled; final position is never dropped", async () => {
  const b = browser();
  const playback = { itemId: "item", responseId: "reply", playedSamples: 6000 };
  b.api.feedback("position", playback);
  assert.equal(b.sent.length, 0);
  await b.api.onEvent({ type: "session.updated", session: { frankie: { playback_feedback: true } } });
  b.api.feedback("position", playback);
  b.clock(100); b.api.feedback("position", playback);
  assert.equal(b.sent.length, 1);
  b.clock(250); b.api.feedback("position", playback);
  b.api.feedback("finished", playback);
  assert.equal(b.sent.length, 3);
  assert.deepEqual(b.sent.at(-1), {
    type: "frankie.playback.finished", item_id: "item", response_id: "reply", audio_end_ms: 250,
  });
});

test("stale response completion cannot mark a newer reply idle", async () => {
  const b = browser();
  await b.api.onEvent({ type: "response.created", response: { id: "new" } });
  await b.api.onEvent({ type: "response.done", response: { id: "old", status: "cancelled" } });
  assert.equal(b.api.active, true);
  await b.api.onEvent({ type: "response.done", response: { id: "new", status: "completed" } });
  assert.equal(b.api.active, false);
});

test("task cancellation is a request, and task names are plain text", async () => {
  const b = browser();
  const task = { call_id: "tool1", name: "<b>Lookup</b>", status: "running" };
  await b.api.onEvent({ type: "frankie.task.updated", task });
  const [label, cancel] = b.get("tasks").children[0].children;
  assert.equal(label.textContent, "<b>Lookup</b> · running");
  cancel.onclick();
  assert.deepEqual(b.sent.at(-1), { type: "frankie.task.cancel", call_id: "tool1" });
  assert.equal(cancel.disabled, true);
  await b.api.onEvent({ type: "frankie.task.updated", task, cancellation_requested: true });
  assert.match(label.textContent, /cancellation requested/);
  assert.equal(cancel.disabled, true);
});

test("demo tool stays disabled unless explicitly enabled", async () => {
  const b = browser(); b.api.settings();
  assert.deepEqual(b.sent.at(-1).session.tools, []);
  await b.api.onEvent({ type: "response.function_call_arguments.done", name: "demo_lookup", call_id: "call", arguments: '{"topic":"weather"}' });
  b.finishTimers();
  assert.equal(b.sent.length, 1);
});

test("duplicate function events cannot run a demo tool twice", async () => {
  const b = browser(); b.get("demoTool").checked = true;
  const event = { type: "response.function_call_arguments.done", name: "demo_lookup", call_id: "call", arguments: '{"topic":"weather"}' };
  await b.api.onEvent(event); await b.api.onEvent(event);
  assert.equal(b.sent.length, 0, "tool completion must be asynchronous");
  b.finishTimers();
  assert.equal(b.sent.length, 2);
  assert.equal(b.sent[0].item.call_id, "call");
  assert.equal(JSON.parse(b.sent[0].item.output).fictional, true);
  assert.equal(b.sent[1].type, "response.create");
});

test("cancelled or superseded tool calls cannot publish results or cancel a replacement", async () => {
  const b = browser(); b.get("demoTool").checked = true;
  for (const call_id of ["old", "new"]) {
    await b.api.onEvent({ type: "response.function_call_arguments.done", name: "demo_lookup", call_id, arguments: '{"topic":"travel"}' });
  }
  await b.api.onEvent({ type: "frankie.task.updated", task: { call_id: "old", name: "demo_lookup", status: "superseded", revision: 1 } });
  await b.api.onEvent({ type: "frankie.task.updated", task: { call_id: "new", name: "demo_lookup", status: "running", revision: 2 } });
  b.finishTimers();
  assert.equal(b.sent.filter((e) => e.type === "conversation.item.create").length, 1);
  assert.equal(b.sent[0].item.call_id, "new");
});

test("cancellation arriving before arguments prevents the delayed call", async () => {
  const b = browser(); b.get("demoTool").checked = true;
  await b.api.onEvent({ type: "frankie.task.updated", task: { call_id: "call", name: "demo_lookup", status: "cancelled", revision: 1 } });
  await b.api.onEvent({ type: "response.function_call_arguments.done", name: "demo_lookup", call_id: "call", arguments: '{"topic":"weather"}' });
  b.finishTimers();
  assert.equal(b.sent.length, 0);
});

test("ending the conversation cancels pending demo timers", async () => {
  const b = browser(); b.get("demoTool").checked = true;
  await b.api.onEvent({ type: "response.function_call_arguments.done", name: "demo_lookup", call_id: "call", arguments: '{"topic":"weather"}' });
  await b.api.end(); b.finishTimers();
  assert.equal(b.sent.length, 0);
});

test("unavailable semantic control is shown rather than claimed", async () => {
  const b = browser();
  await b.api.onEvent({ type: "session.updated", session: { frankie: { background_tasks: true } } });
  assert.match(b.get("features").textContent, /Semantic listener unavailable/);
});

test("machine task notices do not appear as spoken assistant messages", async () => {
  const b = browser();
  await b.api.onEvent({ type: "conversation.item.created", item: {
    id: "task-notice", type: "message", role: "system",
    content: [{ type: "input_text", text: '{"task":"cancelled"}' }],
  } });
  assert.equal(b.get("log").children.length, 0);
  await b.api.onEvent({ type: "conversation.item.created", item: {
    id: "user-message", type: "message", role: "user",
    content: [{ type: "input_text", text: "Cancel that lookup." }],
  } });
  assert.equal(b.get("log").children.length, 1);
});

const audioDelta = (responseId = "reply", itemId = "item", samples = 3072) => ({
  type: "response.output_audio.delta", response_id: responseId, item_id: itemId,
  delta: Buffer.alloc(samples * 2).toString("base64"),
});
const deviceEvent = (type, responseId = "reply", itemId = "item") => ({
  type, playback: {responseId, itemId, playedSamples: 3072},
});

test("generation completion keeps Speaking until the actual worklet drains", async () => {
  const b = browser(), p = player();
  await b.api.onEvent({type: "response.created", response: {id: "reply"}});
  await b.api.onEvent(audioDelta());
  await b.api.onEvent({type: "response.output_audio.done", item_id: "item"});
  await b.api.onEvent({type: "response.done", response: {id: "reply", status: "completed"}});
  assert.equal(b.api.active, false);
  assert.equal(b.get("status").textContent, "Speaking");
  assert.equal(b.get("interaction").textContent, "Listening while speaking");
  for (const message of b.audio) p.send(message);
  p.render(128);
  for (const event of p.events.splice(0)) b.api.onPlaybackState(event);
  assert.equal(b.get("status").textContent, "Speaking");
  p.render(3072);
  for (const event of p.events.splice(0)) b.api.onPlaybackState(event);
  assert.equal(b.get("status").textContent, "Listening");
  assert.equal(b.get("interaction").textContent, "Listening");
});

test("paused queued audio resumes visually and stale controls cannot affect replacement", async () => {
  const b = browser();
  await b.api.onEvent({type: "response.created", response: {id: "reply"}});
  await b.api.onEvent(audioDelta());
  await b.api.onEvent({type: "response.done", response: {id: "reply", status: "completed"}});
  await b.api.onEvent({type: "frankie.playback.pause", response_id: "reply"});
  assert.equal(b.get("status").textContent, "Listening");
  assert.equal(b.get("interaction").textContent, "Listening");
  await b.api.onEvent({type: "frankie.playback.resume", response_id: "reply"});
  assert.equal(b.get("status").textContent, "Speaking");
  await b.api.onEvent({type: "response.created", response: {id: "new"}});
  await b.api.onEvent(audioDelta("new", "next"));
  for (const type of ["finished", "stopped", "paused"]) b.api.onPlaybackState(deviceEvent(type));
  await b.api.onEvent({type: "frankie.playback.clear", response_id: "reply"});
  await b.api.onEvent({type: "frankie.playback.pause", response_id: "reply"});
  assert.equal(b.get("status").textContent, "Speaking");
  assert.equal(b.get("interaction").textContent, "Listening while speaking");
  assert.equal(b.api.active, true);
});

test("clear retires speaking state and late audio cannot visually revive it", async () => {
  const b = browser();
  await b.api.onEvent({type: "response.created", response: {id: "reply"}});
  await b.api.onEvent(audioDelta());
  await b.api.onEvent({type: "frankie.playback.clear", response_id: "reply"});
  assert.equal(b.get("status").textContent, "Listening");
  assert.equal(b.get("interaction").textContent, "Listening");
  await b.api.onEvent(audioDelta());
  b.api.onPlaybackState(deviceEvent("position"));
  assert.equal(b.get("status").textContent, "Listening");
  await b.api.onEvent({type: "response.done", response: {id: "reply", status: "cancelled"}});
  assert.equal(b.get("interaction").textContent, "Listening");
});

test("text-only completion and device drain before response.done keep the correct phase", async () => {
  const b = browser();
  await b.api.onEvent({type: "response.created", response: {id: "text"}});
  await b.api.onEvent({type: "response.done", response: {id: "text", status: "completed"}});
  assert.equal(b.get("status").textContent, "Listening");
  await b.api.onEvent({type: "response.created", response: {id: "reply"}});
  await b.api.onEvent(audioDelta());
  b.api.onPlaybackState(deviceEvent("finished"));
  assert.equal(b.get("status").textContent, "Responding");
  assert.equal(b.get("interaction").textContent, "Listening");
  await b.api.onEvent({type: "response.done", response: {id: "reply", status: "completed"}});
  assert.equal(b.get("status").textContent, "Listening");
});

test("explicit hangup displays Call ended but error cleanup preserves the error", async () => {
  const b = browser();
  await b.api.onEvent({type: "response.created", response: {id: "reply"}});
  await b.api.onEvent(audioDelta());
  await b.get("stop").onclick();
  assert.equal(b.get("status").textContent, "Call ended");
  assert.equal(b.get("interaction").textContent, "Ready when you are");
  assert.equal(b.api.active, false);
  const error = browser();
  error.get("status").textContent = "Connection cannot keep up with live audio.";
  await error.api.end();
  assert.equal(error.get("status").textContent, "Connection cannot keep up with live audio.");
});
