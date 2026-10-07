import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../mtplx/frankie/audio-worklet.mjs", import.meta.url), "utf8");

function player() {
  let Constructor;
  const events = [];
  vm.runInNewContext(source, {
    AudioWorkletProcessor: class {
      constructor() { this.port = {postMessage: e => events.push(e)}; }
    },
    registerProcessor: (_, value) => { Constructor = value; },
    sampleRate: 24000,
  });
  const p = new Constructor();
  const send = data => p.port.onmessage({data});
  const audio = id => send({type: "audio", responseId: id, itemId: `item-${id}`,
                            pcm: new Int16Array(3072).fill(8192)});
  const render = () => {
    const out = new Float32Array(128);
    p.process([[new Float32Array(128)]], [[out]]);
    return out;
  };
  return {p, send, audio, render, events};
}

test("stale clear from interrupted response cannot clear replacement audio", () => {
  const {p, send, audio, render, events} = player();
  audio("old"); render();
  send({type: "clear", responseId: "old", requestId: "interrupt"});
  assert.equal(p.queued, 0);
  assert.equal(events.filter(e => e.type === "stopped").length, 1);
  audio("new");
  const queued = p.queued, played = p.played;
  send({type: "clear", responseId: "old", requestId: "interrupt"});
  assert.equal(p.queued, queued);
  assert.equal(p.played, played);
  assert.equal(p.item.responseId, "new");
  assert.equal(events.filter(e => e.type === "stopped").length, 1);
  assert.ok(render().every(value => value === .25));
  send({type: "clear", responseId: "new", requestId: "interrupt"});
  assert.equal(p.queued, 0);
  assert.equal(p.item, null);
  assert.equal(events.filter(e => e.type === "stopped").length, 2);
  assert.ok(render().every(value => value === 0));
});

test("manual unscoped stop still clears whichever response is playing", () => {
  const {p, send, audio, render, events} = player();
  audio("current"); render();
  send({type: "clear", requestId: "interrupt"});
  assert.equal(p.item, null);
  assert.equal(p.queued, 0);
  assert.equal(events.at(-1).playback.responseId, "current");
  assert.ok(render().every(value => value === 0));
});
