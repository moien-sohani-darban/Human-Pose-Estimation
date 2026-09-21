"""Explicit strict-JSON serialization for the unified pose contract."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any

from ..models import BoundingBox, Keypoint, PersonPose, PoseResult


class PoseSerializationError(ValueError):
    """A unified result cannot be represented as strict protocol JSON."""


def _number(
    value: Real | None,
    field_name: str,
    *,
    optional: bool,
) -> int | float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PoseSerializationError(f"{field_name} must be numeric")
    converted = float(value)
    if not isfinite(converted):
        if optional:
            return None
        raise PoseSerializationError(f"{field_name} must be finite")
    if isinstance(value, int):
        return int(value)
    return converted


def _serialize_keypoint(keypoint: Keypoint) -> dict[str, Any]:
    return {
        "index": keypoint.index,
        "name": keypoint.name,
        "x": _number(keypoint.x, "keypoint.x", optional=False),
        "y": _number(keypoint.y, "keypoint.y", optional=False),
        "z": _number(keypoint.z, "keypoint.z", optional=True),
        "confidence": _number(
            keypoint.confidence,
            "keypoint.confidence",
            optional=True,
        ),
    }


def _serialize_bbox(bbox: BoundingBox | None) -> dict[str, Any] | None:
    if bbox is None:
        return None
    return {
        "x": _number(bbox.x, "bbox.x", optional=False),
        "y": _number(bbox.y, "bbox.y", optional=False),
        "width": _number(bbox.width, "bbox.width", optional=False),
        "height": _number(bbox.height, "bbox.height", optional=False),
    }


def _serialize_person(person: PersonPose) -> dict[str, Any]:
    return {
        "person_id": person.person_id,
        "keypoints": [
            _serialize_keypoint(keypoint) for keypoint in person.keypoints
        ],
        "bbox": _serialize_bbox(person.bbox),
    }


def serialize_pose_result(result: PoseResult) -> dict[str, Any]:
    """Convert a unified result to a stable JSON-compatible dictionary.

    Optional non-finite values are serialized as ``null``. Non-finite required
    coordinates or bounding-box values are rejected rather than emitted as the
    non-standard JSON tokens ``NaN`` or ``Infinity``.
    """
    if not isinstance(result, PoseResult):
        raise PoseSerializationError("result must be a PoseResult")
    backend = (
        result.backend.value
        if hasattr(result.backend, "value")
        else result.backend
    )
    if not isinstance(backend, str) or not backend:
        raise PoseSerializationError("result backend must be a stable string")
    return {
        "success": result.success,
        "backend": backend,
        "people": [_serialize_person(person) for person in result.people],
        "image_width": _number(
            result.image_width,
            "image_width",
            optional=True,
        ),
        "image_height": _number(
            result.image_height,
            "image_height",
            optional=True,
        ),
        "processing_time_ms": _number(
            result.processing_time_ms,
            "processing_time_ms",
            optional=True,
        ),
    }
