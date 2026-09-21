import test from "node:test";
import assert from "node:assert/strict";
import {
  buildEstimateArguments,
  normalizeCommandError,
} from "./protocol.js";

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
