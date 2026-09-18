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
  function render(length=128, input=.125) {
    const out = new Float32Array(length);
    p.process([[new Float32Array(length).fill(input)]], [[out]]);
    return out;
  }
  const audio = (id="a", count=3072) => send({type:"audio", responseId:id, itemId:"item-"+id,
                                             pcm:new Int16Array(count).fill(8192)});
  return {p, send, render, audio, events};
}

test("holding preserves queue and samples while capturing microphone and zero reference", () => {
  const p = player(); p.audio(); p.render(128);
  const played = p.p.played, queued = p.p.queued;
  p.send({type:"capture", enabled:true});
  p.send({type:"pause", responseId:"a"});
  assert.ok(p.render(960).every(value => value === 0));
  assert.equal(p.p.played, played); assert.equal(p.p.queued, queued);
  const captures = p.events.filter(e => e.type === "capture");
  assert.equal(captures.length, 2);
  assert.ok(captures.every(e => e.playback.every(value => value === 0)));
  assert.ok(captures.every(e => e.pcm.every(value => value > 0)));
  assert.ok(!p.events.some(e => e.type === "underrun" || e.type === "finished"));
  p.send({type:"resume", responseId:"a"});
  assert.ok(p.render(128).every(value => value === .25));
  assert.equal(p.p.played, played + 128); assert.equal(p.p.queued, queued - 128);
});

test("generation done cannot discard a paused queue or fake playback finished", () => {
  const p = player(); p.audio(); p.render(128);
  p.send({type:"pause", responseId:"a"});
  p.send({type:"done", itemId:"item-a"});
  assert.ok(p.render(4096).every(value => value === 0));
  assert.equal(p.events.filter(e => e.type === "finished").length, 0);
  p.send({type:"resume", responseId:"a"});
  const out = p.render(4096);
  assert.ok(out.subarray(0, 2944).every(value => value === .25));
  assert.ok(out.subarray(2944).every(value => value === 0));
  const finished = p.events.filter(e => e.type === "finished");
  assert.equal(finished.length, 1);
  assert.equal(finished[0].playback.playedSamples, 3072);
});

test("other-response pause and resume cannot affect active playback", () => {
  const p = player(); p.audio(); p.render(128);
  p.send({type:"pause", responseId:"obsolete"});
  assert.ok(p.render(128).every(value => value === .25));
  p.send({type:"pause", responseId:"a"});
  p.send({type:"resume", responseId:"obsolete"});
  assert.ok(p.render(128).every(value => value === 0));
  p.send({type:"resume", responseId:"a"});
  assert.ok(p.render(128).every(value => value === .25));
});

test("clear discards the hold and stale control cannot pause replacement", () => {
  const p = player(); p.audio(); p.render(128);
  p.send({type:"pause", responseId:"a"});
  p.send({type:"clear", requestId:"interrupt"});
  assert.equal(p.p.queued, 0);
  p.audio("b");
  p.send({type:"pause", responseId:"a"});
  p.send({type:"resume", responseId:"a"});
  assert.ok(p.render(128).every(value => value === .25));
  assert.equal(p.p.item.responseId, "b");
});

test("pause before startup preserves all queued samples until matching resume", () => {
  const p = player(); p.audio("a", 768);
  p.send({type:"pause", responseId:"a"});
  p.audio("a", 3072);
  assert.ok(p.render(128).every(value => value === 0));
  assert.equal(p.p.played, 0); assert.equal(p.p.queued, 3840);
  p.send({type:"resume", responseId:"a"});
  assert.ok(p.render(128).every(value => value === .25));
});
