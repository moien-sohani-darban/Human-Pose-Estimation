# Python Pose Engine

The Python engine exposes a backend-independent `PoseEstimator` interface. Its
`estimate()` method accepts an already-decoded OpenCV BGR `uint8` image with
shape `(height, width, 3)` and returns a unified `PoseResult`; callers do not
import or retain MediaPipe result classes.

## Unified pose schema

- `PoseResult`: `success`, stable `backend` identifier, a list of zero or more
  `people`, and reliable image dimensions and monotonic processing time when
  available.
- `PersonPose`: a deterministic result-local `person_id`, canonical
  `keypoints`, and an optional normalized `bbox`.
- `Keypoint`: stable `index` and `name`, normalized image `x`/`y`, backend
  `z`, and optional `confidence`.
- `BoundingBox`: normalized `x` (left), `y` (top), `width`, and `height`.

Image coordinates use `(0, 0)` at the top-left and `(1, 1)` at the
bottom-right. MediaPipe can report landmarks outside that range; keypoints
preserve those values without clamping. The MediaPipe bounding box is the
smallest axis-aligned box around finite, in-frame landmark coordinates, with
no padding; it is `None` when no such landmark exists.

MediaPipe `z` is preserved as relative, depth-like landmark output. It is not
metric world depth: the hip midpoint is the approximate origin, smaller values
are closer to the camera, and its scale is roughly comparable to `x`.
`confidence` uses MediaPipe landmark `visibility` when finite and available,
otherwise `presence`; it is `None` if neither signal is available. No
confidence is fabricated.

Canonical MediaPipe landmark names, in indices 0 through 32, are:

```text
nose, left_eye_inner, left_eye, left_eye_outer, right_eye_inner, right_eye,
right_eye_outer, left_ear, right_ear, mouth_left, mouth_right, left_shoulder,
right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_pinky,
right_pinky, left_index, right_index, left_thumb, right_thumb, left_hip,
right_hip, left_knee, right_knee, left_ankle, right_ankle, left_heel,
right_heel, left_foot_index, right_foot_index
```

The adapter derives this mapping from MediaPipe 1.0.1's installed
`PoseLandmark` enumeration and verifies the indices in tests.

## MediaPipe Tasks configuration

`MediaPipePoseEstimator` uses MediaPipe 1.0.1's Tasks Pose Landmarker in
`IMAGE` mode. `MediaPipePoseConfig` supports:

- `model_path=None`
- `num_poses=1`
- `min_pose_detection_confidence=0.5`
- `min_pose_presence_confidence=0.5`
- `min_tracking_confidence=0.5`

When `model_path` is omitted, the deterministic default is
`python-engine/models/mediapipe/pose_landmarker.task`. Model assets are not
downloaded automatically and `.task` weights are ignored by Git. A missing
asset raises `ModelAssetNotFoundError` with the resolved expected path; pass an
explicit local path in `MediaPipePoseConfig` to use another location.

The landmarker is initialized once, reused for every `estimate()` call, and
released by `close()` or a `with` block.

## Tests

From `python-engine` with its virtual environment active:

```powershell
python -m pytest
```

Unit tests use fake deterministic landmarker results and require no GPU,
network, or model file. The integration test uses the default local model and
an existing image fixture; it skips cleanly when that model is absent.
