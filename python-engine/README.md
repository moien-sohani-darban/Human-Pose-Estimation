# Python Pose Engine

This package is the local inference engine for Human Pose Estimation. It owns
image decoding at the process boundary, backend lifecycle, result
normalization, and the persistent NDJSON sidecar used by Tauri.

```text
decoded OpenCV image
        ↓
PoseEngine → selected PoseEstimator
        ↓
MediaPipe Tasks or YOLO Pose
        ↓
unified PoseResult
```

Higher-level code never imports or retains backend-native result classes.

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

## Development and tests

Create the repository environment with Python 3.13.5, then install the engine
and test extras:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m integration -ra
```

Unit tests use deterministic fake backends and require no GPU, network, or
model file. Real MediaPipe and YOLO integration tests skip cleanly when their
default local assets are absent.

## Local model assets

The Python package defines one canonical development-time asset location per
supported backend. Paths are resolved from the `python-engine` package root,
so changing the process working directory does not change them:

```text
python-engine/models/mediapipe/pose_landmarker.task
python-engine/models/yolo/yolo11n-pose.pt
```

Model acquisition is always manual. Place a model at its canonical location,
or provide another local file through the backend's `model_path` option (or the
desktop model picker). An explicit path takes precedence over the canonical
default. There is no user-home/cache fallback and no automatic download.
Missing files raise the existing typed model-asset error before either backend
constructor runs.

Model binaries (`.task`, `.pt`, and `.pth`) under `python-engine/models` are
excluded from Git by default to avoid repository growth and because model
distribution terms may differ. The tracked empty directories document where
developers should place assets; no placeholder model is included.

Backend discovery reports filesystem-only metadata for both defaults. An
`exists: true` value means only that a regular file is present; it does not
claim that the file is valid or loadable. Backend initialization remains the
authoritative compatibility check.

Production resource mapping is centralized in `app.runtime`. Development
resolves models from `python-engine`; a PyInstaller build resolves the same
logical paths from its private `_internal` resource root. The current Windows
package intentionally includes no default model weights, so packaged metadata
reports both defaults missing while external custom paths remain supported.

The source-controlled `hpe-python-sidecar.spec` builds a Windows x64 `onedir`
runtime and delegates to the same `app.main.main` protocol implementation.
Build-only packager versions live in the `package` optional dependency group.
Use the repository scripts rather than invoking the spec manually so Python
version checks, offline/autoinstall guards, and the packaged smoke test are
always applied.

## Pose visualization

`PoseVisualizer` renders backend-independent `PoseResult` data onto an
already-decoded OpenCV BGR `uint8` image. It always returns a copy, preserving
the caller's image and all unified pose objects.

```python
from app.utils import PoseVisualizer, PoseVisualizerConfig

visualizer = PoseVisualizer(
    PoseVisualizerConfig(draw_bounding_boxes=True)
)
rendered_image = visualizer.render(image, pose_result)
```

The renderer supports zero, one, or multiple people in deterministic
`PoseResult.people` order. It can draw canonical keypoints and skeleton
connections and can optionally draw each person's normalized bounding box.
The skeleton is defined by the project and has no runtime dependency on
MediaPipe or another pose backend.

Normalized coordinates are converted using
`floor(normalized * image_dimension)`. Exact `1.0` values are capped at the
last valid pixel, so `(1, 1)` maps to `(width - 1, height - 1)`. Points outside
`[0, 1]`, points with non-finite coordinates, and skeleton connections with an
unusable endpoint are skipped. Partially visible bounding boxes are clipped to
the image only for drawing; the normalized contract data is never changed.

`PoseVisualizerConfig` provides explicit keypoint radius and thickness,
skeleton and box thickness, drawing toggles, and colors. Colors are three-item
OpenCV **BGR** tuples with channels from 0 through 255. Keypoint thickness `-1`
means a filled circle; otherwise drawing thicknesses are positive integers.
Bounding-box drawing defaults to off.

`minimum_keypoint_confidence` defaults to `None`, meaning no confidence
filter. When configured, a finite confidence below the threshold is skipped.
A keypoint whose confidence is `None` remains drawable because the unified
contract does not fabricate a missing backend signal.

## YOLO Pose backend

`YoloPoseEstimator` adds multi-person pose estimation through the public
Ultralytics API while preserving the same `PoseEstimator`/`PoseResult`
boundary. It selects the compact official `yolo11n-pose.pt` model for local
development and CPU fallback. The deterministic default location is:

```text
python-engine/models/yolo/yolo11n-pose.pt
```

The repository does not include or download the weights. Obtain
`yolo11n-pose.pt` from the official
[YOLO11 pose documentation](https://docs.ultralytics.com/models/yolo11/), then
place it at that path as a separate, explicit developer action; alternatively,
pass another local path through `YoloPoseConfig(model_path=...)`. The adapter
checks that the file exists before constructing Ultralytics, so imports, unit
tests, and missing-model failures cannot trigger Ultralytics' name-based
automatic download behavior. Project-local `.pt` model assets are ignored by
Git.

Configuration supports `device` (`"cpu"`, `"cuda"`, or `"cuda:N"`),
`confidence_threshold`, `iou_threshold`, and `max_detections`. CPU is the
default; requesting CUDA when PyTorch reports it unavailable raises a typed
configuration error.

The official model uses COCO's 17-keypoint schema, whereas the unified
contract's canonical index space contains 33 MediaPipe-derived positions.
The adapter emits only the 17 observations YOLO actually provides, mapped to
their stable canonical names and indices. It does not synthesize the other 16
keypoints. YOLO has no compatible depth output, so every mapped keypoint has
`z=None`. Confidence comes from Ultralytics `Keypoints.conf`, never from the
person detection score. Bounding boxes use the matching normalized
`Boxes.xyxyn` detection without deriving or padding a replacement.

Unit tests use deterministic in-memory result objects and need no GPU,
network, or weights. The integration test runs on CPU only when the default
local weight file exists; otherwise it skips before estimator construction:

```powershell
python -m pytest
python -m pytest -m integration -ra
```

The installed Ultralytics package publishes AGPL-3.0 and Enterprise licensing
options. Review the official
[Ultralytics licensing information](https://www.ultralytics.com/license) for
the chosen use and distribution scenario; this project documentation makes no
legal conclusion.

## Unified pose engine

`PoseEngine` is the backend-independent entry point for higher-level Python
code. MediaPipe and YOLO are currently available; MediaPipe is the explicit
default. MMPose is a known identifier but remains intentionally unavailable in
the Python 3.13.5 environment because there is no reproducible supported
OpenMMLab/MMCV stack for the current Windows runtime.

```python
from app.core import PoseBackend, PoseEngine, PoseEngineConfig
from app.models.yolo_pose import YoloPoseConfig

config = PoseEngineConfig(
    backend=PoseBackend.YOLO,
    yolo=YoloPoseConfig(model_path="models/yolo/yolo11n-pose.pt"),
)

with PoseEngine(config) as engine:
    result = engine.estimate(image)
```

Each engine instance owns exactly one estimator, selected at construction and
fixed until close. It initializes only that backend, delegates the decoded BGR
image unchanged, and returns the backend's unified `PoseResult` unchanged.
There is no automatic fallback or live backend switching. `close()` is
idempotent, and leaving a context manager closes the selected estimator.

`PoseEngine.available_backends()` returns `mediapipe` and `yolo`; selecting
`mmpose` raises `BackendUnavailableError`, while an unknown name raises
`UnsupportedBackendError`. Backend-specific model paths, thresholds, and
device validation remain owned by their existing configuration classes.

## Python sidecar protocol

Run the persistent protocol-v1 sidecar from `python-engine`:

```powershell
python -m app.main
```

The process reads one UTF-8 JSON object per stdin line and writes exactly one
compact JSON response per stdout line, in request order. Stdout is reserved
for NDJSON protocol responses. Python-level backend output, diagnostics, and
unexpected tracebacks are redirected to stderr. There is no unsolicited
startup message; use `ping` to check readiness.

Requests require an `id` string or integer and a `type`. Omitting
`protocol_version` selects version 1; any supplied version must equal `1`.
The stable envelopes are:

```json
{"id":"req-1","ok":true,"result":{}}
{"id":"req-1","ok":false,"error":{"code":"error_code","message":"..."}}
```

Protocol v1 supports only:

- `ping`: returns `{"status":"ready","protocol_version":1}` without loading a model.
- `get_backends`: reports `mediapipe` and `yolo` as available, `mmpose` as
  unavailable, and additive `models` metadata (`backend`, `model_name`,
  project-relative `default_path` and `display_path`, `exists`,
  and optional `size_bytes`) for the two supported defaults. Protocol version
  remains 1, and discovery does not initialize a model.
- `estimate`: decodes one local `image_path` with OpenCV, delegates the BGR
  `uint8` image through `PoseEngine`, and returns the unified `PoseResult`.
- `estimate_frame`: decodes one base64-encoded in-memory image, delegates the
  resulting OpenCV BGR `uint8` image through the same cached `PoseEngine`, and
  returns the same unified `PoseResult`. This additive command keeps protocol
  version 1.
- `shutdown`: closes every initialized engine, returns a success response,
  then exits. Normal stdin EOF also closes all engines and exits successfully.

Example request:

```json
{"id":"pose-1","type":"estimate","backend":"yolo","image_path":"F:\\Images\\person.jpg","backend_config":{"model_path":"F:\\Models\\yolo11n-pose.pt"}}
```

`estimate` accepts only local image paths; URLs and video paths are not live
inputs. `estimate_frame` accepts strict base64 in `image_base64`, rejects empty,
malformed, undecodable, or decoded payloads larger than 4 MiB, and never writes
temporary files. Webcam capture remains owned by the desktop WebView; Python
only receives individual compressed frames. `backend_config` is optional and
accepts only `model_path`. Relative image and model paths resolve against the
sidecar process working directory. Model files are never downloaded.

Engines are created lazily and the sidecar retains at most one `PoseEngine`
per available backend. The first normalized model-path configuration used for
a backend remains active until shutdown. A later request with a different
configuration returns `invalid_backend_configuration`; it never silently
reconstructs or falls back to another engine.

If a desktop user changes a custom model path after that backend has already
initialized, restart the application before estimating again so the persistent
sidecar can create a new engine with the new configuration.

The serialized pose result includes `success`, `backend`, `people`,
`image_width`, `image_height`, and `processing_time_ms`; people include
`person_id`, `keypoints`, and `bbox`. Optional Python `None` and optional
non-finite numeric values become JSON `null`. Non-finite required coordinates
are rejected, and strict encoding never emits `NaN` or `Infinity`.

## Packaging

The source-controlled `hpe-python-sidecar.spec` builds a Windows x64
PyInstaller `onedir` runtime. The bootstrap imports the same `app.main.main`
entry point used in development, so packaging does not create a second
protocol implementation.

From the repository root:

```powershell
.\python-engine\.venv\Scripts\python.exe -m pip install -e ".\python-engine[package]"
.\scripts\build-python-sidecar.ps1
```

The build script pins and verifies Python/PyInstaller versions, disables pip
index access and Ultralytics auto-installation during analysis, and runs a
packaged import/protocol/frame smoke test. Generated runtimes are written to
`python-engine/dist/` and remain excluded from Git.
