import assert from "node:assert/strict";
import test from "node:test";
import {
  createVideoAnalysisController,
  createVideoSelection,
  formatVideoTime,
} from "./videoPose.js";

function harness({ paused = false, currentTime = 1 } = {}) {
  const media = {
    readyState: 4,
    videoWidth: 1280,
    videoHeight: 720,
    paused,
    ended: false,
    currentTime,
  };
  const scheduled = [];
  const cancelled = [];
  const resolvers = [];
  const captures = [];
  const results = [];
  let invalidations = 0;
  let ended = 0;
  const controller = createVideoAnalysisController({
    media,
    captureCurrentFrame: async () => {
      captures.push(media.currentTime);
      return new Uint8Array([captures.length]);
    },
    estimateFrame: (frame) =>
      new Promise((resolve) => resolvers.push(() => resolve({ frame, people: [] }))),
    onResult: (result, capturedTime) => results.push({ result, capturedTime }),
    onError: assert.fail,
    onInvalidate: () => { invalidations += 1; },
    onEnded: () => { ended += 1; },
    onTiming: () => {},
    schedule: (callback) => { scheduled.push(callback); return callback; },
    cancel: (callback) => cancelled.push(callback),
    now: () => 0,
  });
  return {
    media,
    scheduled,
    cancelled,
    resolvers,
    captures,
    results,
    controller,
    get invalidations() { return invalidations; },
    get ended() { return ended; },
  };
}

async function beginRequest(testHarness) {
  testHarness.controller.start();
  testHarness.scheduled.shift()();
  await Promise.resolve();
  assert.equal(testHarness.controller.isInFlight(), true);
}

async function settleRequest(testHarness) {
  testHarness.resolvers.shift()();
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

test("video selection cancellation preserves state and new selection is normalized", () => {
  const current = { path: "current.mp4" };
  assert.equal(createVideoSelection(current, null, assert.fail), current);
  assert.deepEqual(
    createVideoSelection(current, String.raw`F:\Media\clip.webm`, (path) => `asset://${path}`),
    {
      path: String.raw`F:\Media\clip.webm`,
      name: "clip.webm",
      previewSource: String.raw`asset://F:\Media\clip.webm`,
    },
  );
});

test("analysis waits while paused and resumes with the current playback time", async () => {
  const state = harness({ paused: true, currentTime: 3 });
  state.controller.start();
  assert.equal(state.scheduled.length, 0);
  state.media.currentTime = 8;
  state.media.paused = false;
  state.controller.handlePlay();
  state.scheduled.shift()();
  await Promise.resolve();
  assert.deepEqual(state.captures, [8]);
});

test("only one request is active and a fresh frame follows completion", async () => {
  const state = harness();
  await beginRequest(state);
  assert.deepEqual(state.captures, [1]);
  assert.equal(state.scheduled.length, 0);
  state.media.currentTime = 1.2;
  await settleRequest(state);
  assert.equal(state.results.length, 1);
  assert.equal(state.scheduled.length, 1);
  state.media.currentTime = 1.4;
  state.scheduled.shift()();
  await Promise.resolve();
  assert.deepEqual(state.captures, [1, 1.4]);
});

test("pause stops sampling and ignores the in-flight completion", async () => {
  const state = harness();
  await beginRequest(state);
  state.media.paused = true;
  state.controller.handlePause();
  await settleRequest(state);
  assert.deepEqual(state.results, []);
  assert.equal(state.scheduled.length, 0);
});

test("seek invalidates old result and resumes from the new playing position", async () => {
  const state = harness({ currentTime: 1 });
  await beginRequest(state);
  state.controller.handleSeeking();
  state.media.currentTime = 9;
  await settleRequest(state);
  assert.deepEqual(state.results, []);
  assert.equal(state.invalidations, 1);
  state.controller.handleSeeked();
  state.scheduled.shift()();
  await Promise.resolve();
  assert.deepEqual(state.captures, [1, 9]);
});

test("results beyond the staleness threshold are ignored", async () => {
  const state = harness({ currentTime: 1 });
  await beginRequest(state);
  state.media.currentTime = 2;
  await settleRequest(state);
  assert.deepEqual(state.results, []);
});

test("stop and source replacement prevent late results", async () => {
  for (const action of ["stop", "replaceSource"]) {
    const state = harness();
    await beginRequest(state);
    state.controller[action]();
    await settleRequest(state);
    assert.deepEqual(state.results, []);
    assert.equal(state.scheduled.length, 0);
    assert.equal(state.invalidations, 1);
  }
});

test("video end disables analysis and schedules nothing further", async () => {
  const state = harness();
  await beginRequest(state);
  state.media.ended = true;
  state.media.paused = true;
  state.controller.handleEnded();
  await settleRequest(state);
  assert.equal(state.controller.isEnabled(), false);
  assert.equal(state.ended, 1);
  assert.equal(state.scheduled.length, 0);
});

test("time formatting is stable", () => {
  assert.equal(formatVideoTime(0), "0:00");
  assert.equal(formatVideoTime(65.9), "1:05");
  assert.equal(formatVideoTime(Number.NaN), "0:00");
});
