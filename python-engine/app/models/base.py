"""Backend-independent pose estimation contracts and interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


ImageArray = NDArray[np.uint8]


def _require_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


@dataclass(frozen=True, slots=True)
class Keypoint:
    """One canonical body landmark.

    ``x`` and ``y`` use image-normalized coordinates, where the top-left is
    ``(0, 0)`` and the bottom-right is ``(1, 1)``. Backends may report
    out-of-frame values; they are preserved rather than silently clamped.
    ``z`` and ``confidence`` are optional because their semantics and
    availability vary by backend.
    """

    index: int
    name: str
    x: float
    y: float
    z: float | None
    confidence: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("keypoint index must be a non-negative integer")
        if not self.name:
            raise ValueError("keypoint name must not be empty")
        _require_finite(self.x, "keypoint x")
        _require_finite(self.y, "keypoint y")
        if self.z is not None:
            _require_finite(self.z, "keypoint z")
        if self.confidence is not None:
            _require_finite(self.confidence, "keypoint confidence")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Smallest normalized axis-aligned box containing valid landmarks."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "width", "height"):
            _require_finite(getattr(self, field_name), f"bounding box {field_name}")
        if self.x < 0.0 or self.y < 0.0:
            raise ValueError("bounding box origin must be within the image")
        if self.width < 0.0 or self.height < 0.0:
            raise ValueError("bounding box dimensions must be non-negative")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("bounding box must fit within normalized image bounds")


@dataclass(slots=True)
class PersonPose:
    """A detected person's canonical keypoints and optional bounding box."""

    person_id: int
    keypoints: list[Keypoint]
    bbox: BoundingBox | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.person_id, bool)
            or not isinstance(self.person_id, int)
            or self.person_id < 0
        ):
            raise ValueError("person_id must be a non-negative integer")
        self.keypoints = list(self.keypoints)
        indices = [keypoint.index for keypoint in self.keypoints]
        if len(indices) != len(set(indices)):
            raise ValueError("keypoint indices must be unique within a person")


@dataclass(slots=True)
class PoseResult:
    """Backend-neutral result returned by every pose estimator."""

    success: bool
    backend: str
    people: list[PersonPose]
    image_width: int | None = None
    image_height: int | None = None
    processing_time_ms: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        if not self.backend:
            raise ValueError("backend must not be empty")
        self.people = list(self.people)
        person_ids = [person.person_id for person in self.people]
        if len(person_ids) != len(set(person_ids)):
            raise ValueError("person IDs must be unique within a result")
        if (self.image_width is None) != (self.image_height is None):
            raise ValueError("image width and height must be provided together")
        if self.image_width is not None:
            if (
                self.image_width <= 0
                or self.image_height is None
                or self.image_height <= 0
            ):
                raise ValueError("image dimensions must be positive")
        if self.processing_time_ms is not None:
            _require_finite(self.processing_time_ms, "processing_time_ms")
            if self.processing_time_ms < 0.0:
                raise ValueError("processing_time_ms must be non-negative")


class PoseEstimator(ABC):
    """Backend-independent interface implemented by every pose estimator."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Return the stable backend identifier."""
        raise NotImplementedError

    @abstractmethod
    def estimate(self, image: ImageArray) -> PoseResult:
        """Estimate poses in an already-decoded OpenCV-compatible image."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release backend resources; repeated calls must be safe."""
        raise NotImplementedError

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
