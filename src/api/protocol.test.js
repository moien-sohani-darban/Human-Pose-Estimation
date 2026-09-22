import test from "node:test";
import assert from "node:assert/strict";
import {
  buildEstimateArguments,
  buildEstimateFrameArguments,
  normalizeCommandError,
} from "./protocol.js";
import { selectVideoFile, VIDEO_FILTER } from "./pose.js";

test("buildEstimateArguments matches the Task 08 command shape", () => {
  assert.deepEqual(
    buildEstimateArguments({
      backend: "yolo",
      imagePath: String.raw`C:\Images\person.jpg`,
      modelPath: String.raw`C:\Models\pose.pt`,
    }),
    {
      request: {
        backend: "yolo",
        imagePath: String.raw`C:\Images\person.jpg`,
        modelPath: String.raw`C:\Models\pose.pt`,
      },
    },
  );
});

test("buildEstimateFrameArguments carries bytes without browser base64", () => {
  assert.deepEqual(
    buildEstimateFrameArguments({
      backend: "mediapipe",
      frameData: new Uint8Array([255, 216, 255]),
      modelPath: "",
    }),
    {
      request: {
        backend: "mediapipe",
        frameData: [255, 216, 255],
        modelPath: null,
      },
    },
  );
});

test("empty model paths become null without changing the image path", () => {
  const imagePath = String.raw`F:\People\subject one.png`;

  const payload = buildEstimateArguments({
    backend: "mediapipe",
    imagePath,
    modelPath: "  ",
  });

  assert.equal(payload.request.imagePath, imagePath);
  assert.equal(payload.request.modelPath, null);
});

test("packaged runtime errors become actionable without exposing paths", () => {
  assert.deepEqual(
    normalizeCommandError({
      code: "sidecar_runtime_missing",
      message: String.raw`C:\Users\developer\internal\python-sidecar.exe`,
    }),
    {
      code: "sidecar_runtime_missing",
      message:
        "The packaged pose engine is unavailable. Reinstall the application and try again.",
    },
  );
});

test("structured bridge errors preserve codes and become actionable", () => {
  assert.deepEqual(
    normalizeCommandError({
      kind: "python_protocol_error",
      code: "model_asset_not_found",
      message: "machine path",
    }),
    {
      code: "model_asset_not_found",
      message:
        "This backend needs a compatible local model. Select one with Browse and try again.",
    },
  );
});

test("serialized and unsafe unknown errors are normalized safely", () => {
  assert.equal(
    normalizeCommandError(
      JSON.stringify({ kind: "sidecar_timeout", message: "late" }),
    ).code,
    "sidecar_timeout",
  );

  assert.deepEqual(normalizeCommandError("Traceback\nsecret path"), {
    code: "unexpected_error",
    message:
      "Something unexpected happened. Check your input and try again.",
  });
});

test("unmapped typed errors never expose backend diagnostics", () => {
  assert.deepEqual(
    normalizeCommandError({
      code: "future_internal_failure",
      message: String.raw`F:\private\engine.py: traceback detail`,
    }),
    {
      code: "future_internal_failure",
      message:
        "Something unexpected happened. Check your input and try again.",
    },
  );
});

test("video picker uses conservative WebView formats and cancellation is normal", async () => {
  let options;

  const selected = await selectVideoFile(async (received) => {
    options = received;
    return null;
  });

  assert.equal(selected, null);
  assert.deepEqual(options.filters, [VIDEO_FILTER]);
  assert.deepEqual(VIDEO_FILTER.extensions, ["mp4", "webm", "mov", "m4v"]);
  assert.equal(options.multiple, false);
  assert.equal(options.directory, false);
});
