import assert from "node:assert/strict";
import test from "node:test";
import {
  computeInferenceDimensions,
  canvasToJpegBytes,
  captureMediaFrame,
  createLiveScheduler,
  normalizeCameraError,
  requestCamera,
  stopMediaStream,
} from "./livePose.js";

test("capture dimensions preserve aspect ratio and cap the longest side", () => {
  assert.deepEqual(computeInferenceDimensions(1920, 1080), { width: 960, height: 540 });
  assert.deepEqual(computeInferenceDimensions(1280, 720), { width: 960, height: 540 });
  assert.deepEqual(computeInferenceDimensions(720, 1280), { width: 540, height: 960 });
  assert.deepEqual(computeInferenceDimensions(640, 480), { width: 640, height: 480 });
});

test("shared media capture reuses canvas context and bounded geometry", async () => {
  const draws = [];
  const context = { drawImage: (...args) => draws.push(args) };
  const canvas = {
    width: 0,
    height: 0,
    getContext: () => context,
    toBlob: (callback) => callback(new Blob([new Uint8Array([4, 5])])),
  };
  const media = { readyState: 4, videoWidth: 1920, videoHeight: 1080 };

  const first = await captureMediaFrame(media, canvas);
  const second = await captureMediaFrame(media, canvas, first.context);

  assert.deepEqual([first.width, first.height], [960, 540]);
  assert.deepEqual([...second.frameData], [4, 5]);
  assert.equal(first.context, context);
  assert.equal(second.context, context);
  assert.equal(draws.length, 2);
  assert.deepEqual(draws[0].slice(1), [0, 0, 960, 540]);
});

test("JPEG capture returns bytes and enforces the payload ceiling", async () => {
  const bytes = await canvasToJpegBytes({
    toBlob(callback, type, quality) {
      assert.equal(type, "image/jpeg");
      assert.equal(quality, 0.75);
      callback(new Blob([new Uint8Array([1, 2, 3])], { type }));
    },
  });
  assert.deepEqual([...bytes], [1, 2, 3]);

  await assert.rejects(
    canvasToJpegBytes({
      toBlob(callback) {
        callback({ size: 4 * 1024 * 1024 + 1 });
      },
    }),
    (error) => error.code === "frame_too_large",
  );
});

test("camera request uses video only and stop releases every track", async () => {
  const tracks = [{ stopped: 0, stop() { this.stopped += 1; } }, { stopped: 0, stop() { this.stopped += 1; } }];
  const stream = { getTracks: () => tracks };
  let constraints;
  const returned = await requestCamera({ getUserMedia: async (value) => { constraints = value; return stream; } });
  assert.equal(returned, stream);
  assert.equal(constraints.audio, false);
  assert.equal(constraints.video.width.ideal, 1280);
  stopMediaStream(stream);
  assert.deepEqual(tracks.map((track) => track.stopped), [1, 1]);
});

test("camera errors are actionable", () => {
  assert.equal(normalizeCameraError({ name: "NotAllowedError" }).code, "camera_permission_denied");
  assert.equal(normalizeCameraError({ name: "NotFoundError" }).code, "camera_not_found");
  assert.equal(normalizeCameraError({ name: "NotReadableError" }).code, "camera_in_use");
});

test("missing mediaDevices reports camera unavailable", async () => {
  await assert.rejects(
    requestCamera(undefined),
    (error) => error.code === "camera_unavailable",
  );
});

test("scheduler permits one in-flight request and captures fresh after completion", async () => {
  const scheduled = [];
  const captures = [];
  const resolvers = [];
  const results = [];
  const scheduler = createLiveScheduler({
    captureFrame: async () => { const frame = captures.length + 1; captures.push(frame); return frame; },
    estimateFrame: (frame) => new Promise((resolve) => resolvers.push(() => resolve({ frame }))),
    onResult: (result) => results.push(result),
    onError: assert.fail,
    onTiming: () => {},
    schedule: (callback) => { scheduled.push(callback); return callback; },
    cancel: () => {},
    now: () => 0,
  });
  scheduler.start();
  scheduled.shift()();
  await Promise.resolve();
  scheduler.requestFreshFrame();
  scheduler.requestFreshFrame();
  assert.deepEqual(captures, [1]);
  assert.equal(scheduler.isInFlight(), true);
  resolvers.shift()();
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(results, [{ frame: 1 }]);
  scheduled.shift()();
  await Promise.resolve();
  assert.deepEqual(captures, [1, 2]);
  scheduler.stop();
});

test("stopping invalidates an in-flight completion", async () => {
  const scheduled = [];
  let resolveEstimate;
  const results = [];
  const scheduler = createLiveScheduler({
    captureFrame: async () => new Uint8Array([1]),
    estimateFrame: () => new Promise((resolve) => { resolveEstimate = resolve; }),
    onResult: (result) => results.push(result),
    onError: assert.fail,
    onTiming: () => {},
    schedule: (callback) => { scheduled.push(callback); return callback; },
    cancel: () => {},
  });
  scheduler.start();
  scheduled.shift()();
  await Promise.resolve();
  scheduler.stop();
  resolveEstimate({ people: [{ person_id: 1 }] });
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(results, []);
  assert.equal(scheduler.isActive(), false);
});
