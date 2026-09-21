"""Deterministic unit tests for the Ultralytics YOLO Pose adapter."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ultralytics.engine.results import Boxes, Keypoints, Results

from app.models import (
    BoundingBox,
    EstimatorClosedError,
    InvalidInputImageError,
    PoseEstimator,
    PoseResult,
)
from app.models import yolo_pose
from app.models.yolo_pose import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    YOLO_COCO_KEYPOINT_MAPPING,
    YoloDependencyUnavailableError,
    YoloInferenceError,
    YoloInitializationError,
    YoloInvalidConfigurationError,
    YoloMalformedResultError,
    YoloModelAssetNotFoundError,
    YoloPoseConfig,
    YoloPoseEstimator,
)
from app.utils import PoseVisualizer, PoseVisualizerConfig


EXPECTED_MAPPING = (
    (0, 0, "nose"),
    (1, 2, "left_eye"),
    (2, 5, "right_eye"),
    (3, 7, "left_ear"),
    (4, 8, "right_ear"),
    (5, 11, "left_shoulder"),
    (6, 12, "right_shoulder"),
    (7, 13, "left_elbow"),
    (8, 14, "right_elbow"),
    (9, 15, "left_wrist"),
    (10, 16, "right_wrist"),
    (11, 23, "left_hip"),
    (12, 24, "right_hip"),
    (13, 25, "left_knee"),
    (14, 26, "right_knee"),
    (15, 27, "left_ankle"),
    (16, 28, "right_ankle"),
)
UNSUPPORTED_NAMES = {
    "left_eye_inner",
    "left_eye_outer",
    "mouth_left",
    "left_pinky",
    "left_heel",
    "left_foot_index",
}


class FakeYoloModel:
    task = "pose"
    model = SimpleNamespace(kpt_shape=(17, 3))

    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        return self.results


def native_result(person_count: int = 1) -> SimpleNamespace:
    coordinates = np.empty((person_count, 17, 2), dtype=np.float64)
    confidence = np.empty((person_count, 17), dtype=np.float64)
    boxes = np.empty((person_count, 4), dtype=np.float64)
    for person_index in range(person_count):
        for keypoint_index in range(17):
            coordinates[person_index, keypoint_index] = (
                0.01 * (keypoint_index + 1) + 0.1 * person_index,
                0.02 * (keypoint_index + 1) + 0.1 * person_index,
            )
            confidence[person_index, keypoint_index] = (
                0.51 + 0.02 * keypoint_index - 0.1 * person_index
            )
        boxes[person_index] = (
            0.1 + 0.1 * person_index,
            0.2 + 0.1 * person_index,
            0.5 + 0.1 * person_index,
            0.8 + 0.1 * person_index,
        )
    return SimpleNamespace(
        keypoints=SimpleNamespace(xyn=coordinates, conf=confidence),
        boxes=SimpleNamespace(xyxyn=boxes),
    )


def create_estimator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    results: list[object],
) -> tuple[YoloPoseEstimator, FakeYoloModel]:
    model_path = tmp_path / DEFAULT_MODEL_NAME
    model_path.touch()
    fake_model = FakeYoloModel(results)
    monkeypatch.setattr(YoloPoseEstimator, "_create_model", lambda self: fake_model)
    estimator = YoloPoseEstimator(YoloPoseConfig(model_path=model_path))
    return estimator, fake_model


def test_installed_ultralytics_result_api_boundary() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = np.array([[10, 5, 50, 45, 0.9, 0]], dtype=np.float32)
    keypoints = np.ones((1, 17, 3), dtype=np.float32)
    result = Results(image, "fixture.jpg", {0: "person"}, boxes=boxes, keypoints=keypoints)

    assert isinstance(result.boxes, Boxes)
    assert isinstance(result.keypoints, Keypoints)
    assert result.boxes.xyxyn.shape == (1, 4)
    assert result.keypoints.xyn.shape == (1, 17, 2)
    assert result.keypoints.conf.shape == (1, 17)


def test_complete_coco_mapping_uses_canonical_names_and_indices() -> None:
    assert YOLO_COCO_KEYPOINT_MAPPING == EXPECTED_MAPPING
    assert [item[0] for item in YOLO_COCO_KEYPOINT_MAPPING] == list(range(17))
    assert UNSUPPORTED_NAMES.isdisjoint(
        item[2] for item in YOLO_COCO_KEYPOINT_MAPPING
    )


def test_adapter_maps_all_keypoints_without_fabrication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native_result()])

    result = estimator.estimate(np.zeros((100, 200, 3), dtype=np.uint8))
    keypoints = result.people[0].keypoints

    assert isinstance(estimator, PoseEstimator)
    assert isinstance(result, PoseResult)
    assert result.backend == "yolo"
    assert len(keypoints) == 17
    assert [(point.index, point.name) for point in keypoints] == [
        (canonical_index, name)
        for _, canonical_index, name in EXPECTED_MAPPING
    ]
    assert all(point.z is None for point in keypoints)
    assert all(point.name not in UNSUPPORTED_NAMES for point in keypoints)
    assert keypoints[0].x == pytest.approx(0.01)
    assert keypoints[16].y == pytest.approx(0.34)


def test_keypoint_confidence_uses_per_keypoint_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = native_result()
    native.boxes.conf = np.array([0.01])
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native])

    result = estimator.estimate(np.zeros((20, 20, 3), dtype=np.uint8))

    assert [point.confidence for point in result.people[0].keypoints] == pytest.approx(
        [0.51 + 0.02 * index for index in range(17)]
    )
    assert all(point.confidence != 0.01 for point in result.people[0].keypoints)


def test_missing_keypoint_confidence_maps_to_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = native_result()
    native.keypoints.conf = None
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native])

    result = estimator.estimate(np.zeros((20, 20, 3), dtype=np.uint8))

    assert all(point.confidence is None for point in result.people[0].keypoints)


def test_yolo_box_normalizes_to_unified_bbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native_result()])

    result = estimator.estimate(np.zeros((100, 200, 3), dtype=np.uint8))

    bbox = result.people[0].bbox
    assert bbox is not None
    assert (bbox.x, bbox.y, bbox.width, bbox.height) == pytest.approx(
        (0.1, 0.2, 0.4, 0.6)
    )


def test_invalid_or_missing_boxes_are_not_fabricated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = native_result()
    native.boxes.xyxyn[0] = (-0.1, 0.2, 0.5, 0.8)
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native])
    result = estimator.estimate(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result.people[0].bbox is None

    native = native_result()
    native.boxes = None
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native])
    result = estimator.estimate(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result.people[0].bbox is None


def test_multiple_people_preserve_order_and_associations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native_result(2)])

    result = estimator.estimate(np.zeros((100, 200, 3), dtype=np.uint8))

    assert [person.person_id for person in result.people] == [0, 1]
    assert result.people[0].keypoints[0].x == pytest.approx(0.01)
    assert result.people[1].keypoints[0].x == pytest.approx(0.11)
    first_bbox = result.people[0].bbox
    second_bbox = result.people[1].bbox
    assert first_bbox is not None
    assert second_bbox is not None
    assert (
        first_bbox.x,
        first_bbox.y,
        first_bbox.width,
        first_bbox.height,
    ) == pytest.approx((0.1, 0.2, 0.4, 0.6))
    assert (
        second_bbox.x,
        second_bbox.y,
        second_bbox.width,
        second_bbox.height,
    ) == pytest.approx((0.2, 0.3, 0.4, 0.6))


@pytest.mark.parametrize(
    "native",
    [
        SimpleNamespace(keypoints=None, boxes=None),
        SimpleNamespace(
            keypoints=SimpleNamespace(
                xyn=np.empty((0, 17, 2)), conf=np.empty((0, 17))
            ),
            boxes=SimpleNamespace(xyxyn=np.empty((0, 4))),
        ),
    ],
)
def test_empty_detection_returns_empty_people(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, native: object
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native])

    result = estimator.estimate(np.zeros((5, 6, 3), dtype=np.uint8))

    assert result.people == []
    assert (result.image_width, result.image_height) == (6, 5)


@pytest.mark.parametrize(
    "image",
    [
        None,
        "not an array",
        np.array([], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
    ],
)
def test_invalid_input_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image: object
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native_result()])

    with pytest.raises(InvalidInputImageError):
        estimator.estimate(image)  # type: ignore[arg-type]


def test_default_model_path_is_project_local() -> None:
    assert DEFAULT_MODEL_PATH.name == "yolo11n-pose.pt"
    assert DEFAULT_MODEL_PATH.parent.name == "yolo"
    assert DEFAULT_MODEL_PATH.parents[1].name == "models"


def test_missing_model_path_raises_typed_error(tmp_path: Path) -> None:
    missing = tmp_path / DEFAULT_MODEL_NAME

    with pytest.raises(YoloModelAssetNotFoundError) as raised:
        YoloPoseEstimator(YoloPoseConfig(model_path=missing))

    assert raised.value.expected_path == missing.resolve()
    assert str(missing.resolve()) in str(raised.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_path": "  "},
        {"device": "gpu"},
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"confidence_threshold": float("nan")},
        {"iou_threshold": True},
        {"max_detections": 0},
        {"max_detections": 1.5},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(YoloInvalidConfigurationError):
        YoloPoseConfig(**kwargs)  # type: ignore[arg-type]


def test_unavailable_cuda_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / DEFAULT_MODEL_NAME
    model_path.touch()
    monkeypatch.setattr(yolo_pose, "_cuda_is_available", lambda: False)

    with pytest.raises(YoloInvalidConfigurationError, match="CUDA is unavailable"):
        YoloPoseEstimator(YoloPoseConfig(model_path=model_path, device="cuda:0"))


def test_dependency_unavailable_is_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / DEFAULT_MODEL_NAME
    model_path.touch()
    monkeypatch.setattr(yolo_pose, "_UltralyticsYOLO", None)
    monkeypatch.setattr(
        yolo_pose, "_ULTRALYTICS_IMPORT_ERROR", ImportError("missing dependency")
    )

    with pytest.raises(YoloDependencyUnavailableError) as raised:
        YoloPoseEstimator(YoloPoseConfig(model_path=model_path))

    assert isinstance(raised.value.__cause__, ImportError)


def test_initialization_failure_preserves_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / DEFAULT_MODEL_NAME
    model_path.touch()

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("invalid weights")

    monkeypatch.setattr(yolo_pose, "_UltralyticsYOLO", fail)
    with pytest.raises(YoloInitializationError, match="invalid weights") as raised:
        YoloPoseEstimator(YoloPoseConfig(model_path=model_path))
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_non_pose_or_non_coco_model_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / DEFAULT_MODEL_NAME
    model_path.touch()
    wrong_model = SimpleNamespace(task="detect", model=SimpleNamespace(kpt_shape=None))
    monkeypatch.setattr(YoloPoseEstimator, "_create_model", lambda self: wrong_model)
    with pytest.raises(YoloInitializationError, match="expected 'pose'"):
        YoloPoseEstimator(YoloPoseConfig(model_path=model_path))

    wrong_model = SimpleNamespace(task="pose", model=SimpleNamespace(kpt_shape=(33, 3)))
    monkeypatch.setattr(YoloPoseEstimator, "_create_model", lambda self: wrong_model)
    with pytest.raises(YoloInitializationError, match="17-keypoint"):
        YoloPoseEstimator(YoloPoseConfig(model_path=model_path))


def test_prediction_configuration_and_model_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, model = create_estimator(monkeypatch, tmp_path, [native_result()])
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    first = estimator.estimate(image)
    second = estimator.estimate(image)

    assert first.processing_time_ms is not None
    assert second.processing_time_ms is not None
    assert len(model.calls) == 2
    assert model.calls[0]["source"] is image
    assert {key: value for key, value in model.calls[0].items() if key != "source"} == {
        "device": "cpu",
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "verbose": False,
    }


def test_inference_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, model = create_estimator(monkeypatch, tmp_path, [native_result()])

    def fail(**kwargs: object) -> list[object]:
        raise RuntimeError("prediction failed")

    model.predict = fail  # type: ignore[method-assign]
    with pytest.raises(YoloInferenceError, match="prediction failed") as raised:
        estimator.estimate(np.zeros((10, 10, 3), dtype=np.uint8))
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "native",
    [
        SimpleNamespace(
            keypoints=SimpleNamespace(xyn=np.zeros((1, 16, 2)), conf=None),
            boxes=None,
        ),
        SimpleNamespace(
            keypoints=SimpleNamespace(
                xyn=np.zeros((1, 17, 2)), conf=np.zeros((1, 16))
            ),
            boxes=None,
        ),
        SimpleNamespace(
            keypoints=SimpleNamespace(xyn=np.zeros((1, 17, 2)), conf=None),
            boxes=SimpleNamespace(xyxyn=np.zeros((2, 4))),
        ),
        SimpleNamespace(
            keypoints=None,
            boxes=SimpleNamespace(xyxyn=np.zeros((1, 4))),
        ),
    ],
)
def test_malformed_results_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, native: object
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native])

    with pytest.raises(YoloMalformedResultError):
        estimator.estimate(np.zeros((10, 10, 3), dtype=np.uint8))


@pytest.mark.parametrize("results", [[], [native_result(), native_result()]])
def test_result_count_must_match_single_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, results: list[object]
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, results)

    with pytest.raises(YoloMalformedResultError, match="exactly one result"):
        estimator.estimate(np.zeros((10, 10, 3), dtype=np.uint8))


def test_close_is_idempotent_and_prevents_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native_result()])
    estimator.close()
    estimator.close()

    with pytest.raises(EstimatorClosedError):
        estimator.estimate(np.zeros((10, 10, 3), dtype=np.uint8))


def test_yolo_unified_result_is_visualizer_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    estimator, _ = create_estimator(monkeypatch, tmp_path, [native_result()])
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = estimator.estimate(image)

    rendered = PoseVisualizer(
        PoseVisualizerConfig(draw_bounding_boxes=True)
    ).render(image, result)

    assert rendered.shape == image.shape
    assert rendered.dtype == image.dtype
    assert np.any(rendered)
