import test from "node:test";
import assert from "node:assert/strict";
import {
  buildRenderablePeople,
  CANONICAL_SKELETON,
  fileNameFromPath,
  formatPeopleCount,
  resolveSkeletonSegments,
  setBackendModelPath,
  validatePoseResult,
} from "./pose.js";

const point = (index, name, x, y) => ({
  index,
  name,
  x,
  y,
  z: null,
  confidence: 0.9,
});

test("canonical skeleton mirrors the 35 Task 03 connections", () => {
  assert.equal(CANONICAL_SKELETON.length, 35);
  assert.deepEqual(CANONICAL_SKELETON.at(-1), [
    "right_ankle",
    "right_foot_index",
  ]);
});

test("sparse YOLO-like keypoints create only observed canonical edges", () => {
  const person = {
    keypoints: [
      point(11, "left_shoulder", 0.2, 0.3),
      point(13, "left_elbow", 0.3, 0.5),
      point(12, "right_shoulder", 0.7, 0.3),
    ],
  };

  const segments = resolveSkeletonSegments(person);

  assert.deepEqual(
    segments.map(({ startName, endName }) => [startName, endName]),
    [
      ["left_shoulder", "right_shoulder"],
      ["left_shoulder", "left_elbow"],
    ],
  );
});

test("multiple people retain distinct points, colors, and optional bboxes", () => {
  const people = buildRenderablePeople({
    people: [
      {
        person_id: 2,
        keypoints: [point(0, "nose", 0.5, 0.1)],
        bbox: { x: 0.2, y: 0.1, width: 0.4, height: 0.8 },
      },
      {
        person_id: 7,
        keypoints: [point(0, "nose", 1.2, -0.1)],
        bbox: null,
      },
    ],
  });

  assert.equal(people.length, 2);
  assert.notEqual(people[0].color, people[1].color);
  assert.deepEqual(people[0].bbox, {
    x: 0.2,
    y: 0.1,
    width: 0.4,
    height: 0.8,
  });
  assert.equal(people[1].bbox, null);
  assert.equal(people[1].keypoints[0].x, 1.2);
});

test("empty and malformed people arrays remain safe", () => {
  assert.deepEqual(buildRenderablePeople({ people: [] }), []);
  assert.deepEqual(buildRenderablePeople({}), []);
});

test("result and display helpers are deterministic", () => {
  assert.equal(
    validatePoseResult({ success: true, backend: "yolo", people: [] }),
    true,
  );
  assert.equal(validatePoseResult({ backend: "yolo" }), false);
  assert.equal(
    fileNameFromPath(String.raw`C:\Images\person.jpg`),
    "person.jpg",
  );
  assert.equal(formatPeopleCount(1), "1 person");
  assert.equal(formatPeopleCount(3), "3 people");
});

test("separate backend model paths remain independent", () => {
  const initial = { mediapipe: "", yolo: "" };
  const withMediaPipe = setBackendModelPath(
    initial,
    "mediapipe",
    "pose.task",
  );
  const withYolo = setBackendModelPath(withMediaPipe, "yolo", "pose.pt");

  assert.deepEqual(withYolo, {
    mediapipe: "pose.task",
    yolo: "pose.pt",
  });
  assert.deepEqual(initial, { mediapipe: "", yolo: "" });
});
