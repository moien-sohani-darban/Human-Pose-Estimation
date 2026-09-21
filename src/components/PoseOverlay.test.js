import test from "node:test";
import assert from "node:assert/strict";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

test("PoseOverlay renders deterministic multi-person and sparse fixture geometry", async () => {
  const vite = await createServer({
    server: { middlewareMode: true },
    appType: "custom",
  });

  try {
    const { default: PoseOverlay } = await vite.ssrLoadModule(
      "/src/components/PoseOverlay.jsx",
    );

    const result = {
      success: true,
      backend: "yolo",
      image_width: 800,
      image_height: 600,
      processing_time_ms: 8.2,
      people: [
        {
          person_id: 0,
          keypoints: [
            {
              index: 11,
              name: "left_shoulder",
              x: 0.25,
              y: 0.3,
            },
            {
              index: 13,
              name: "left_elbow",
              x: 0.2,
              y: 0.5,
            },
            {
              index: 12,
              name: "right_shoulder",
              x: 0.75,
              y: 0.3,
            },
          ],
          bbox: {
            x: 0.1,
            y: 0.1,
            width: 0.8,
            height: 0.85,
          },
        },
        {
          person_id: 1,
          keypoints: [
            {
              index: 0,
              name: "nose",
              x: 1.1,
              y: -0.1,
            },
          ],
          bbox: null,
        },
      ],
    };

    const markup = renderToStaticMarkup(
      PoseOverlay({
        result,
        showSkeleton: true,
        showKeypoints: true,
        showBoxes: true,
      }),
    );

    assert.equal((markup.match(/class="pose-line"/g) ?? []).length, 2);
    assert.equal((markup.match(/class="pose-point"/g) ?? []).length, 4);
    assert.equal((markup.match(/class="pose-box"/g) ?? []).length, 1);
    assert.match(markup, /x1="25%"/);
    assert.match(markup, /cx="110/);
  } finally {
    await vite.close();
  }
});

test("PoseOverlay renders no primitives for a zero-person success", async () => {
  const vite = await createServer({
    server: { middlewareMode: true },
    appType: "custom",
  });

  try {
    const { default: PoseOverlay } = await vite.ssrLoadModule(
      "/src/components/PoseOverlay.jsx",
    );

    const markup = renderToStaticMarkup(
      PoseOverlay({
        result: {
          success: true,
          backend: "mediapipe",
          people: [],
        },
        showSkeleton: true,
        showKeypoints: true,
        showBoxes: true,
      }),
    );

    assert.doesNotMatch(markup, /<(line|circle|rect)/);
  } finally {
    await vite.close();
  }
});
