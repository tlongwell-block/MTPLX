// Exercise the real AudioWorklet, with a recorded microphone fixture paced live.
import { parseArgs } from "node:util";
const { values } = parseArgs({
  options: {
    url: { type: "string", default: "http://127.0.0.1:18870" },
    audio: { type: "string" },
    "long-audio": { type: "string" },
    output: {
      type: "string",
      default: "outputs/frankie-browser-" + Date.now(),
    },
  },
});
if (!values.audio || !process.env.MTPLX_FRANKIE_TOKEN)
  throw new Error("Supply --audio and MTPLX_FRANKIE_TOKEN");
const { chromium } = await import(
  process.env.FRANKIE_PLAYWRIGHT_MODULE || "playwright"
);
import fs from "node:fs";
import assert from "node:assert/strict";
const config = { token: process.env.MTPLX_FRANKIE_TOKEN };
const output = values.output;
fs.mkdirSync(output, { recursive: true });
function pcm(file) {
  const b = fs.readFileSync(file);
  let p = 12;
  while (b.toString("ascii", p, p + 4) !== "data")
    p += 8 + b.readUInt32LE(p + 4) + (b.readUInt32LE(p + 4) % 2);
  return Array.from({ length: b.readUInt32LE(p + 4) / 2 }, (_, i) =>
    b.readInt16LE(p + 8 + i * 2),
  );
}
const arithmetic = pcm(values.audio);
const story = values["long-audio"] ? pcm(values["long-audio"]) : null;
const browser = await chromium.launch({
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage({ viewport: { width: 1120, height: 960 } });
page.setDefaultTimeout(90000);
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.addInitScript(() => {
  navigator.mediaDevices.getUserMedia = async () => {
    const c = new AudioContext({ sampleRate: 24000 }),
      d = c.createMediaStreamDestination();
    await c.resume();
    window.probeSpeak = async (samples) => {
      const b = c.createBuffer(1, samples.length, 24000);
      b.copyToChannel(
        Float32Array.from(samples, (x) => x / 32768),
        0,
      );
      const s = c.createBufferSource();
      s.buffer = b;
      s.connect(d);
      window.frankieTest.evidence.push({
        type: "fixture_start",
        time: performance.now(),
      });
      s.start();
      await new Promise((r) => (s.onended = r));
      window.frankieTest.evidence.push({
        type: "fixture_end",
        time: performance.now(),
      });
    };
    return d.stream;
  };
});
const results = [];
async function before() {
  return page.evaluate(() => window.frankieTest.evidence.length);
}
async function events(n) {
  return page.evaluate((n) => window.frankieTest.evidence.slice(n), n);
}
async function complete(n, label, word) {
  await page.waitForFunction(
    (n) =>
      window.frankieTest.evidence
        .slice(n)
        .some((e) => e.type === "response.done" || e.type === "error"),
    n,
  );
  await page.waitForFunction((n) => {
    const es = window.frankieTest.evidence.slice(n),
      totals = {};
    for (const e of es)
      if (e.type === "audio_received")
        totals[e.itemId] = (totals[e.itemId] || 0) + e.samples;
    return (
      Object.keys(totals).length &&
      Object.entries(totals).every(([id, total]) =>
        es.some(
          (e) =>
            e.type === "playback" &&
            e.itemId === id &&
            e.playedSamples >= total,
        ),
      )
    );
  }, n);
  const es = await events(n);
  assert.equal(
    es.filter((e) => e.type === "error").length,
    0,
    JSON.stringify(es),
  );
  const text = await page.locator("#log .message").last().innerText();
  if (word) assert.match(text, word);
  const first = es.find((e) => e.type === "audio_received"),
    end = es.find((e) => e.type === "fixture_end");
  assert.equal(
    es.filter((e) => e.type === "underrun").length,
    0,
    "playback underrun",
  );
  const result = {
    label,
    text,
    audioSeconds:
      es
        .filter((e) => e.type === "audio_received")
        .reduce((n, e) => n + e.samples, 0) / 24000,
    captureFrames: es.filter((e) => e.type === "capture").length,
    underruns: es.filter((e) => e.type === "underrun").length,
    firstAudioAfterFixtureMs: first && end ? first.time - end.time : null,
  };
  results.push(result);
  console.log(JSON.stringify(result));
}
try {
  await page.goto(values.url + "/#" + config.token);
  await page.locator("#start").click();
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Listening",
  );
  let n = await before();
  await page.evaluate((x) => window.probeSpeak(x), arithmetic);
  await complete(n, "voice arithmetic", /four|4/i);
  n = await before();
  await page.evaluate((x) => window.probeSpeak(x), arithmetic);
  await complete(n, "repeated voice arithmetic", /four|4/i);
  n = await before();
  await page
    .locator("#text")
    .fill("Describe a peaceful garden in ten sentences.");
  await page.locator("#send").click();
  await page.waitForFunction(
    (n) =>
      window.frankieTest.evidence
        .slice(n)
        .some((e) => e.type === "audio_received"),
    n,
  );
  await page.evaluate((x) => window.probeSpeak(x), arithmetic);
  await page.waitForFunction(
    (n) =>
      window.frankieTest.evidence
        .slice(n)
        .some((e) => e.type === "playback_stopped"),
    n,
  );
  await page.waitForFunction(
    (n) =>
      window.frankieTest.evidence
        .slice(n)
        .filter((e) => e.type === "response.done").length >= 2,
    n,
  );
  let es = await events(n);
  assert.equal(
    es.filter((e) => e.type === "error").length,
    0,
    JSON.stringify(es),
  );
  results.push({
    label: "barge-in",
    stops: es.filter((e) => e.type === "playback_stopped").length,
    text: await page.locator("#log").innerText(),
  });
  console.log(JSON.stringify(results.at(-1)));
  await page.waitForTimeout(800);
  n = await before();
  if (story) {
    await page.evaluate((x) => window.probeSpeak(x), story);
  } else {
    await page
      .locator("#text")
      .fill("Describe a peaceful garden in ten sentences.");
    await page.locator("#send").click();
  }
  await complete(n, "long speech", story ? /garden/i : undefined);
  if (!story)
    assert.ok(
      results.at(-1).audioSeconds > 10,
      "expected sustained spoken output",
    );
  await page.screenshot({ path: output + "/demo.png", fullPage: true });
  await page.locator("#stop").click();
  await page.locator("#start").click();
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Listening",
  );
  n = await before();
  await page.evaluate((x) => window.probeSpeak(x), arithmetic);
  await complete(n, "reconnect", /four|4/i);
  assert.equal(errors.length, 0, errors.join("\n"));
  console.log("PASS " + output);
} finally {
  fs.writeFileSync(
    output + "/results.json",
    JSON.stringify({ results, errors }, null, 2),
  );
  fs.writeFileSync(
    output + "/events.json",
    JSON.stringify(
      await page.evaluate(() => window.frankieTest?.evidence || []),
      null,
      2,
    ),
  );
  await page
    .screenshot({ path: output + "/final.png", fullPage: true })
    .catch(() => {});
  await browser.close();
}
