"""MediaPipe Tasks Pose Landmarker adapter for the unified pose contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .base import (
    BoundingBox,
    ImageArray,
    Keypoint,
    PersonPose,
    PoseEstimator,
    PoseResult,
)
from .assets import get_default_model_path
from .errors import (
    BackendInferenceError,
    BackendInitializationError,
    BackendResourceError,
    EstimatorClosedError,
    InvalidConfigurationError,
    InvalidInputImageError,
    ModelAssetNotFoundError,
)


BACKEND_ID = "mediapipe"
DEFAULT_MODEL_PATH = get_default_model_path(BACKEND_ID)


def _installed_landmark_names() -> tuple[str, ...]:
    """Build the canonical order from the installed Tasks API enumeration."""
    members = sorted(mp.tasks.vision.PoseLandmark, key=int)
    indices = tuple(int(member) for member in members)
    if indices != tuple(range(len(members))):
        raise RuntimeError("MediaPipe PoseLandmark indices are not contiguous")
    return tuple(member.name.lower() for member in members)


MEDIAPIPE_POSE_KEYPOINT_NAMES = _installed_landmark_names()


class MediaPipeInitializationError(BackendInitializationError):
    """MediaPipe could not create its reusable Pose Landmarker."""


class MediaPipeInferenceError(BackendInferenceError):
    """MediaPipe inference or result normalization failed."""


class MediaPipeResourceError(BackendResourceError):
    """MediaPipe could not release its Pose Landmarker cleanly."""


@dataclass(frozen=True, slots=True)
class MediaPipePoseConfig:
    """Supported MediaPipe Pose Landmarker options for IMAGE mode."""

    model_path: str | Path | None = None
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.num_poses, bool) or not isinstance(self.num_poses, int):
            raise InvalidConfigurationError("num_poses must be a positive integer")
        if self.num_poses <= 0:
            raise InvalidConfigurationError("num_poses must be a positive integer")

        for field_name in (
            "min_pose_detection_confidence",
            "min_pose_presence_confidence",
            "min_tracking_confidence",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidConfigurationError(
                    f"{field_name} must be a number between 0 and 1"
                )
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise InvalidConfigurationError(
                    f"{field_name} must be between 0 and 1"
                )

        if self.model_path is not None:
            if isinstance(self.model_path, str) and not self.model_path.strip():
                raise InvalidConfigurationError("model_path must not be empty")
            try:
                Path(self.model_path)
            except TypeError as error:
                raise InvalidConfigurationError(
                    "model_path must be a path-like value"
                ) from error


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve an explicit path or the documented project-local default."""
    try:
        candidate = (
            get_default_model_path(BACKEND_ID)
            if model_path is None
            else Path(model_path)
        )
    except TypeError as error:
        raise InvalidConfigurationError("model_path must be a path-like value") from error

    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise ModelAssetNotFoundError(resolved)
    return resolved


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if isfinite(converted) else None


def _normalize_landmarker_result(
    result: Any,
    *,
    image_width: int,
    image_height: int,
    processing_time_ms: float,
) -> PoseResult:
    """Convert one native PoseLandmarkerResult without exposing it to callers."""
    native_poses = getattr(result, "pose_landmarks", None)
    if native_poses is None:
        raise MediaPipeInferenceError("MediaPipe result has no pose_landmarks field")

    people: list[PersonPose] = []
    for person_id, native_landmarks in enumerate(native_poses):
        if len(native_landmarks) != len(MEDIAPIPE_POSE_KEYPOINT_NAMES):
            raise MediaPipeInferenceError(
                "MediaPipe returned an unexpected landmark count: "
                f"expected {len(MEDIAPIPE_POSE_KEYPOINT_NAMES)}, "
                f"received {len(native_landmarks)}"
            )

        keypoints: list[Keypoint] = []
        box_points: list[tuple[float, float]] = []
        for index, (name, landmark) in enumerate(
            zip(MEDIAPIPE_POSE_KEYPOINT_NAMES, native_landmarks, strict=True)
        ):
            try:
                x = float(landmark.x)
                y = float(landmark.y)
            except (AttributeError, TypeError, ValueError) as error:
                raise MediaPipeInferenceError(
                    f"MediaPipe landmark {index} has invalid x/y coordinates"
                ) from error
            if not isfinite(x) or not isfinite(y):
                raise MediaPipeInferenceError(
                    f"MediaPipe landmark {index} has non-finite x/y coordinates"
                )

            try:
                z = _optional_finite_float(getattr(landmark, "z", None))
                visibility = _optional_finite_float(
                    getattr(landmark, "visibility", None)
                )
                presence = _optional_finite_float(getattr(landmark, "presence", None))
            except (TypeError, ValueError, OverflowError) as error:
                raise MediaPipeInferenceError(
                    f"MediaPipe landmark {index} has invalid optional values"
                ) from error

            confidence = visibility if visibility is not None else presence
            keypoints.append(Keypoint(index, name, x, y, z, confidence))
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                box_points.append((x, y))

        bbox = None
        if box_points:
            xs, ys = zip(*box_points, strict=True)
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            bbox = BoundingBox(left, top, right - left, bottom - top)

        people.append(PersonPose(person_id, keypoints, bbox))

    return PoseResult(
        success=True,
        backend=BACKEND_ID,
        people=people,
        image_width=image_width,
        image_height=image_height,
        processing_time_ms=processing_time_ms,
    )


class MediaPipePoseEstimator(PoseEstimator):
    """Reusable MediaPipe Tasks Pose Landmarker running in IMAGE mode."""

    def __init__(self, config: MediaPipePoseConfig | None = None) -> None:
        self.config = config or MediaPipePoseConfig()
        if not isinstance(self.config, MediaPipePoseConfig):
            raise InvalidConfigurationError(
                "config must be a MediaPipePoseConfig instance"
            )
        self.model_path = resolve_model_path(self.config.model_path)
        self._landmarker = self._create_landmarker()

    @property
    def backend(self) -> str:
        return BACKEND_ID

    def _create_landmarker(self) -> Any:
        try:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=self.config.num_poses,
                min_pose_detection_confidence=(
                    self.config.min_pose_detection_confidence
                ),
                min_pose_presence_confidence=(
                    self.config.min_pose_presence_confidence
                ),
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            return mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except (OSError, RuntimeError, ValueError) as error:
            raise MediaPipeInitializationError(
                f"Failed to initialize MediaPipe Pose Landmarker: {error}"
            ) from error

    def estimate(self, image: ImageArray) -> PoseResult:
        if self._landmarker is None:
            raise EstimatorClosedError("MediaPipe estimator has already been closed")
        if not isinstance(image, np.ndarray):
            raise InvalidInputImageError("image must be a NumPy array")
        if image.size == 0:
            raise InvalidInputImageError("image must not be empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidInputImageError("image must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise InvalidInputImageError("image dtype must be uint8")

        image_height, image_width = image.shape[:2]
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            media_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_image),
            )
        except (cv2.error, RuntimeError, ValueError) as error:
            raise InvalidInputImageError(
                f"image could not be converted for MediaPipe: {error}"
            ) from error

        started = monotonic()
        try:
            native_result = self._landmarker.detect(media_image)
        except (OSError, RuntimeError, ValueError) as error:
            raise MediaPipeInferenceError(
                f"MediaPipe Pose Landmarker inference failed: {error}"
            ) from error
        elapsed_ms = (monotonic() - started) * 1000.0
        return _normalize_landmarker_result(
            native_result,
            image_width=image_width,
            image_height=image_height,
            processing_time_ms=elapsed_ms,
        )

    def close(self) -> None:
        landmarker = self._landmarker
        if landmarker is None:
            return
        self._landmarker = None
        try:
            landmarker.close()
        except (OSError, RuntimeError, ValueError) as error:
            raise MediaPipeResourceError(
                f"Failed to close MediaPipe Pose Landmarker: {error}"
            ) from error
