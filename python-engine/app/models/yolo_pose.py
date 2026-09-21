"""Ultralytics YOLO Pose adapter for the unified pose contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from re import fullmatch
from time import monotonic
from typing import Any

import numpy as np

from .assets import get_default_model_path, get_model_asset_definition

from .base import (
    BoundingBox,
    ImageArray,
    Keypoint,
    PersonPose,
    PoseEstimator,
    PoseResult,
)
from .errors import (
    BackendInferenceError,
    BackendInitializationError,
    EstimatorClosedError,
    InvalidConfigurationError,
    InvalidInputImageError,
    ModelAssetNotFoundError,
    PoseEstimationError,
)

try:
    from ultralytics import YOLO as _UltralyticsYOLO
except ImportError as error:  # pragma: no cover - exercised through monkeypatching
    _UltralyticsYOLO = None
    _ULTRALYTICS_IMPORT_ERROR: ImportError | None = error
else:
    _ULTRALYTICS_IMPORT_ERROR = None


BACKEND_ID = "yolo"
DEFAULT_MODEL_NAME = get_model_asset_definition(BACKEND_ID).relative_path.name
DEFAULT_MODEL_PATH = get_default_model_path(BACKEND_ID)

# (YOLO index, canonical project index, canonical project name)
YOLO_COCO_KEYPOINT_MAPPING: tuple[tuple[int, int, str], ...] = (
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


class YoloDependencyUnavailableError(PoseEstimationError):
    """The Ultralytics/PyTorch dependency boundary is unavailable."""


class YoloModelAssetNotFoundError(ModelAssetNotFoundError):
    """The configured local YOLO pose weight file does not exist."""

    def __init__(self, expected_path: Path) -> None:
        self.expected_path = expected_path
        PoseEstimationError.__init__(
            self,
            "YOLO pose model asset is missing. "
            f"Expected a local file at: {expected_path}. "
            "Pass model_path in YoloPoseConfig to configure another location.",
        )


class YoloInvalidConfigurationError(InvalidConfigurationError):
    """YOLO backend configuration is invalid or unsupported."""


class YoloInitializationError(BackendInitializationError):
    """Ultralytics could not initialize the local pose model."""


class YoloInferenceError(BackendInferenceError):
    """Ultralytics failed while running pose inference."""


class YoloMalformedResultError(YoloInferenceError):
    """Ultralytics returned data that cannot map safely to the contract."""


@dataclass(frozen=True, slots=True)
class YoloPoseConfig:
    """Supported Ultralytics prediction settings for decoded images."""

    model_path: str | Path | None = None
    device: str = "cpu"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    max_detections: int = 300

    def __post_init__(self) -> None:
        if self.model_path is not None:
            if isinstance(self.model_path, str) and not self.model_path.strip():
                raise YoloInvalidConfigurationError("model_path must not be empty")
            try:
                Path(self.model_path)
            except TypeError as error:
                raise YoloInvalidConfigurationError(
                    "model_path must be a path-like value"
                ) from error

        if not isinstance(self.device, str) or fullmatch(
            r"cpu|cuda(?::\d+)?", self.device.strip().lower()
        ) is None:
            raise YoloInvalidConfigurationError(
                "device must be 'cpu', 'cuda', or 'cuda:<index>'"
            )
        object.__setattr__(self, "device", self.device.strip().lower())

        for field_name in ("confidence_threshold", "iou_threshold"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise YoloInvalidConfigurationError(
                    f"{field_name} must be a finite number between 0 and 1"
                )

        if (
            isinstance(self.max_detections, bool)
            or not isinstance(self.max_detections, int)
            or self.max_detections <= 0
        ):
            raise YoloInvalidConfigurationError(
                "max_detections must be a positive integer"
            )


def resolve_yolo_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve a configured path or the deterministic project-local default."""
    try:
        candidate = (
            get_default_model_path(BACKEND_ID)
            if model_path is None
            else Path(model_path)
        )
    except TypeError as error:
        raise YoloInvalidConfigurationError(
            "model_path must be a path-like value"
        ) from error
    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise YoloModelAssetNotFoundError(resolved)
    return resolved


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency is installed normally
        raise YoloDependencyUnavailableError(
            "PyTorch is unavailable; reinstall the Task 04 dependencies"
        ) from error
    return bool(torch.cuda.is_available())


def _to_numpy(value: object, field_name: str) -> np.ndarray:
    """Copy a NumPy or CPU-convertible tensor into adapter-owned memory."""
    try:
        converted = value.detach() if hasattr(value, "detach") else value
        converted = converted.cpu() if hasattr(converted, "cpu") else converted
        converted = converted.numpy() if hasattr(converted, "numpy") else converted
        return np.asarray(converted).copy()
    except (RuntimeError, TypeError, ValueError) as error:
        raise YoloMalformedResultError(
            f"YOLO result field {field_name} cannot be converted to NumPy"
        ) from error


def _valid_bbox(values: np.ndarray) -> BoundingBox | None:
    try:
        left, top, right, bottom = (float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(value) for value in (left, top, right, bottom)):
        return None
    if right < left or bottom < top:
        return None
    if not all(0.0 <= value <= 1.0 for value in (left, top, right, bottom)):
        return None
    return BoundingBox(left, top, right - left, bottom - top)


def _normalize_yolo_result(
    native_result: Any,
    *,
    image_width: int,
    image_height: int,
    processing_time_ms: float,
) -> PoseResult:
    native_keypoints = getattr(native_result, "keypoints", None)
    native_boxes = getattr(native_result, "boxes", None)

    if native_keypoints is None:
        if native_boxes is None:
            return PoseResult(
                True,
                BACKEND_ID,
                [],
                image_width,
                image_height,
                processing_time_ms,
            )
        try:
            box_count = len(native_boxes)
        except TypeError as error:
            raise YoloMalformedResultError(
                "YOLO boxes must provide a sized detection collection"
            ) from error
        if box_count == 0:
            return PoseResult(
                True,
                BACKEND_ID,
                [],
                image_width,
                image_height,
                processing_time_ms,
            )
        raise YoloMalformedResultError(
            "YOLO returned person boxes without pose keypoints"
        )

    normalized_xy = _to_numpy(
        getattr(native_keypoints, "xyn", None), "keypoints.xyn"
    )
    if normalized_xy.ndim != 3 or normalized_xy.shape[1:] != (17, 2):
        raise YoloMalformedResultError(
            "YOLO keypoints.xyn must have shape (people, 17, 2)"
        )
    person_count = normalized_xy.shape[0]

    native_confidence = getattr(native_keypoints, "conf", None)
    confidence: np.ndarray | None = None
    if native_confidence is not None:
        confidence = _to_numpy(native_confidence, "keypoints.conf")
        if confidence.shape != (person_count, 17):
            raise YoloMalformedResultError(
                "YOLO keypoints.conf must have shape (people, 17)"
            )

    boxes: np.ndarray | None = None
    if native_boxes is not None:
        boxes = _to_numpy(getattr(native_boxes, "xyxyn", None), "boxes.xyxyn")
        if boxes.shape != (person_count, 4):
            raise YoloMalformedResultError(
                "YOLO boxes.xyxyn must align one-to-one with pose keypoints"
            )

    people: list[PersonPose] = []
    for person_id in range(person_count):
        keypoints: list[Keypoint] = []
        for yolo_index, canonical_index, canonical_name in YOLO_COCO_KEYPOINT_MAPPING:
            x = float(normalized_xy[person_id, yolo_index, 0])
            y = float(normalized_xy[person_id, yolo_index, 1])
            if not isfinite(x) or not isfinite(y):
                raise YoloMalformedResultError(
                    f"YOLO keypoint {yolo_index} has non-finite coordinates"
                )
            keypoint_confidence = None
            if confidence is not None:
                candidate = float(confidence[person_id, yolo_index])
                if isfinite(candidate):
                    keypoint_confidence = candidate
            keypoints.append(
                Keypoint(
                    index=canonical_index,
                    name=canonical_name,
                    x=x,
                    y=y,
                    z=None,
                    confidence=keypoint_confidence,
                )
            )

        bbox = None if boxes is None else _valid_bbox(boxes[person_id])
        people.append(PersonPose(person_id, keypoints, bbox))

    return PoseResult(
        True,
        BACKEND_ID,
        people,
        image_width,
        image_height,
        processing_time_ms,
    )


class YoloPoseEstimator(PoseEstimator):
    """Reusable Ultralytics YOLO Pose estimator for decoded BGR images."""

    def __init__(self, config: YoloPoseConfig | None = None) -> None:
        if config is not None and not isinstance(config, YoloPoseConfig):
            raise YoloInvalidConfigurationError(
                "config must be a YoloPoseConfig instance"
            )
        self.config = config or YoloPoseConfig()
        self.model_path = resolve_yolo_model_path(self.config.model_path)
        if self.config.device.startswith("cuda") and not _cuda_is_available():
            raise YoloInvalidConfigurationError(
                f"device '{self.config.device}' requested but CUDA is unavailable"
            )
        self._model = self._create_model()
        self._validate_model_schema()

    @property
    def backend(self) -> str:
        return BACKEND_ID

    def _create_model(self) -> Any:
        if _UltralyticsYOLO is None:
            raise YoloDependencyUnavailableError(
                "Ultralytics is unavailable; install the Task 04 dependencies"
            ) from _ULTRALYTICS_IMPORT_ERROR
        try:
            return _UltralyticsYOLO(self.model_path, task="pose", verbose=False)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise YoloInitializationError(
                f"Failed to initialize YOLO pose model: {error}"
            ) from error

    def _validate_model_schema(self) -> None:
        task = getattr(self._model, "task", None)
        if task is not None and task != "pose":
            raise YoloInitializationError(
                f"Configured YOLO model task is '{task}', expected 'pose'"
            )
        model = getattr(self._model, "model", None)
        keypoint_shape = getattr(model, "kpt_shape", None)
        if keypoint_shape is not None:
            try:
                uses_coco_pose_schema = (
                    len(keypoint_shape) >= 1 and int(keypoint_shape[0]) == 17
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise YoloInitializationError(
                    "Configured YOLO model has an invalid keypoint schema"
                ) from error
            if not uses_coco_pose_schema:
                raise YoloInitializationError(
                    "Configured YOLO pose model must use the COCO 17-keypoint schema"
                )

    def estimate(self, image: ImageArray) -> PoseResult:
        if self._model is None:
            raise EstimatorClosedError("YOLO estimator has already been closed")
        if not isinstance(image, np.ndarray):
            raise InvalidInputImageError("image must be a NumPy array")
        if image.size == 0:
            raise InvalidInputImageError("image must not be empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidInputImageError("image must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise InvalidInputImageError("image dtype must be uint8")

        image_height, image_width = image.shape[:2]
        started = monotonic()
        try:
            native_results = self._model.predict(
                source=image,
                device=self.config.device,
                conf=float(self.config.confidence_threshold),
                iou=float(self.config.iou_threshold),
                max_det=self.config.max_detections,
                verbose=False,
            )
            native_results = list(native_results)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise YoloInferenceError(f"YOLO pose inference failed: {error}") from error
        elapsed_ms = (monotonic() - started) * 1000.0

        if len(native_results) != 1:
            raise YoloMalformedResultError(
                "YOLO inference for one image must return exactly one result"
            )
        return _normalize_yolo_result(
            native_results[0],
            image_width=image_width,
            image_height=image_height,
            processing_time_ms=elapsed_ms,
        )

    def close(self) -> None:
        self._model = None

