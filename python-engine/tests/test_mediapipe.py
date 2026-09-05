"""Deterministic unit tests for the MediaPipe Tasks adapter."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import pytest

from app.models import (
    BoundingBox,
    EstimatorClosedError,
    InvalidConfigurationError,
    InvalidInputImageError,
    ModelAssetNotFoundError,
    PoseEstimator,
    PoseResult,
)
from app.models.mediapipe_pose import (
    MEDIAPIPE_POSE_KEYPOINT_NAMES,
    MediaPipeInferenceError,
    MediaPipeInitializationError,
    MediaPipePoseConfig,
    MediaPipePoseEstimator,
)


FIXTURE_DIRECTORY = Path(__file__).parent
FIXTURE_NAMES = ("test-person.jfif", "test_person.jfif")
EXPECTED_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


class FakeLandmarker:
    def __init__(self, result: object) -> None:
        self.result = result
        self.detect_calls = 0
        self.closed = False

    def detect(self, image: mp.Image) -> object:
        assert isinstance(image, mp.Image)
        self.detect_calls += 1
        return self.result

    def close(self) -> None:
        self.closed = True


def landmark(
    index: int,
    *,
    visibility: float | None = 0.8,
    presence: float | None = 0.7,
) -> SimpleNamespace:
    return SimpleNamespace(
        x=0.1 + index * 0.02,
        y=0.2 + index * 0.01,
        z=-0.01 * index,
        visibility=visibility,
        presence=presence,
    )


def pose_result(*poses: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(pose_landmarks=list(poses))


def create_estimator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: object,
) -> tuple[MediaPipePoseEstimator, FakeLandmarker]:
    model_path = tmp_path / "pose_landmarker.task"
    model_path.touch()
    fake = FakeLandmarker(result)
    monkeypatch.setattr(MediaPipePoseEstimator, "_create_landmarker", lambda self: fake)
    estimator = MediaPipePoseEstimator(MediaPipePoseConfig(model_path=model_path))
    return estimator, fake


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_local_image_fixture_is_readable(fixture_name: str) -> None:
    image = cv2.imread(str(FIXTURE_DIRECTORY / fixture_name))

    assert image is not None
    assert image.shape == (547, 365, 3)


def test_installed_mediapipe_exposes_tasks_only_api() -> None:
    assert mp.__version__ == "1.0.1"
    assert hasattr(mp, "tasks")
    assert not hasattr(mp, "solutions")


def test_canonical_mapping_comes_from_installed_enum() -> None:
    installed = tuple(
        item.name.lower() for item in sorted(mp.tasks.vision.PoseLandmark, key=int)
    )

    assert MEDIAPIPE_POSE_KEYPOINT_NAMES == EXPECTED_NAMES
    assert MEDIAPIPE_POSE_KEYPOINT_NAMES == installed
    assert tuple(int(item) for item in sorted(mp.tasks.vision.PoseLandmark, key=int)) == tuple(
        range(33)
    )


def test_adapter_maps_multiple_people_confidence_and_bbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = [landmark(index) for index in range(33)]
    first[0] = landmark(0, visibility=None, presence=0.61)
    first[1].x = -0.25
    first[2].y = 1.25
    second = [landmark(index, visibility=0.55) for index in range(33)]
    estimator, fake = create_estimator(
        monkeypatch, tmp_path, pose_result(first, second)
    )

    result = estimator.estimate(np.zeros((20, 40, 3), dtype=np.uint8))

    assert isinstance(estimator, PoseEstimator)
    assert isinstance(result, PoseResult)
    assert estimator.backend == "mediapipe"
    assert [person.person_id for person in result.people] == [0, 1]
    assert [point.index for point in result.people[0].keypoints] == list(range(33))
    assert tuple(point.name for point in result.people[0].keypoints) == EXPECTED_NAMES
    assert result.people[0].keypoints[0].confidence == 0.61
    assert result.people[0].keypoints[3].confidence == 0.8
    assert result.people[0].keypoints[1].x == -0.25
    assert result.people[0].bbox == BoundingBox(0.1, 0.2, 0.64, 0.32)
    assert (result.image_width, result.image_height) == (40, 20)
    assert result.processing_time_ms is not None
    assert fake.detect_calls == 1


def test_empty_native_result_maps_to_empty_unified_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, pose_result())

    result = estimator.estimate(np.zeros((2, 3, 3), dtype=np.uint8))

    assert result.people == []
    assert result.success is True


def test_bbox_is_none_when_all_landmarks_are_out_of_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native_pose = [landmark(index) for index in range(33)]
    for point in native_pose:
        point.x = -1.0
    estimator, _ = create_estimator(monkeypatch, tmp_path, pose_result(native_pose))

    result = estimator.estimate(np.zeros((2, 3, 3), dtype=np.uint8))

    assert result.people[0].bbox is None


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.array([], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
    ],
)
def test_invalid_input_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image: object
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, pose_result())

    with pytest.raises(InvalidInputImageError):
        estimator.estimate(image)  # type: ignore[arg-type]


def test_missing_model_reports_expected_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.task"

    with pytest.raises(ModelAssetNotFoundError) as raised:
        MediaPipePoseEstimator(MediaPipePoseConfig(model_path=missing))

    assert raised.value.expected_path == missing.resolve()
    assert str(missing.resolve()) in str(raised.value)


def test_initialization_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "invalid.task"
    model_path.touch()

    def fail(_: object) -> object:
        raise RuntimeError("invalid model data")

    monkeypatch.setattr(
        mp.tasks.vision.PoseLandmarker, "create_from_options", fail
    )
    with pytest.raises(MediaPipeInitializationError, match="invalid model data"):
        MediaPipePoseEstimator(MediaPipePoseConfig(model_path=model_path))


def test_initialization_uses_image_mode_and_supported_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.task"
    model_path.touch()
    captured: dict[str, object] = {}
    fake = FakeLandmarker(pose_result())

    def capture(options: object) -> FakeLandmarker:
        captured["options"] = options
        return fake

    monkeypatch.setattr(
        mp.tasks.vision.PoseLandmarker, "create_from_options", capture
    )
    config = MediaPipePoseConfig(
        model_path=model_path,
        num_poses=3,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.7,
    )
    estimator = MediaPipePoseEstimator(config)
    options = captured["options"]

    assert options.running_mode is mp.tasks.vision.RunningMode.IMAGE
    assert options.num_poses == 3
    assert options.min_pose_detection_confidence == 0.4
    assert options.min_pose_presence_confidence == 0.6
    assert options.min_tracking_confidence == 0.7
    assert options.output_segmentation_masks is False
    estimator.close()


def test_blank_model_path_is_invalid_configuration() -> None:
    with pytest.raises(InvalidConfigurationError, match="must not be empty"):
        MediaPipePoseConfig(model_path="  ")


@pytest.mark.parametrize("value", [0, -1, True, 1.1])
def test_invalid_num_poses_is_rejected(value: object) -> None:
    with pytest.raises(InvalidConfigurationError):
        MediaPipePoseConfig(num_poses=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True])
def test_invalid_confidence_configuration_is_rejected(value: object) -> None:
    with pytest.raises(InvalidConfigurationError):
        MediaPipePoseConfig(min_pose_detection_confidence=value)  # type: ignore[arg-type]


def test_inference_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, fake = create_estimator(monkeypatch, tmp_path, pose_result())

    def fail(_: mp.Image) -> object:
        raise RuntimeError("native failure")

    fake.detect = fail  # type: ignore[method-assign]
    with pytest.raises(MediaPipeInferenceError, match="native failure"):
        estimator.estimate(np.zeros((2, 2, 3), dtype=np.uint8))


def test_landmarker_is_reused_and_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, fake = create_estimator(monkeypatch, tmp_path, pose_result())
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    estimator.estimate(image)
    estimator.estimate(image)
    estimator.close()
    estimator.close()

    assert fake.detect_calls == 2
    assert fake.closed is True
    with pytest.raises(EstimatorClosedError):
        estimator.estimate(image)
