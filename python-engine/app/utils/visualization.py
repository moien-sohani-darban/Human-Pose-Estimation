"""Backend-independent OpenCV rendering for the unified pose contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import cv2
import numpy as np

from ..models import BoundingBox, Keypoint, PoseResult


Color = tuple[int, int, int]
Pixel = tuple[int, int]
SkeletonConnection = tuple[str, str]


CANONICAL_KEYPOINT_NAMES = frozenset(
    {
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
    }
)


CANONICAL_SKELETON: tuple[SkeletonConnection, ...] = (
    ("nose", "left_eye_inner"),
    ("left_eye_inner", "left_eye"),
    ("left_eye", "left_eye_outer"),
    ("left_eye_outer", "left_ear"),
    ("nose", "right_eye_inner"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("right_eye_outer", "right_ear"),
    ("mouth_left", "mouth_right"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_pinky"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_thumb"),
    ("left_pinky", "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_pinky"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_thumb"),
    ("right_pinky", "right_index"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
)


class VisualizationError(Exception):
    """Base class for predictable visualization failures."""


class InvalidVisualizationInputError(VisualizationError):
    """The image or unified pose result cannot be rendered safely."""


class InvalidVisualizationConfigError(VisualizationError):
    """A renderer configuration value is invalid."""


class VisualizationRenderingError(VisualizationError):
    """OpenCV failed while drawing an otherwise valid render request."""


def _validate_color(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise InvalidVisualizationConfigError(
            f"{field_name} must be a three-integer BGR tuple"
        )
    if any(
        isinstance(channel, bool)
        or not isinstance(channel, int)
        or not 0 <= channel <= 255
        for channel in value
    ):
        raise InvalidVisualizationConfigError(
            f"{field_name} channels must be integers between 0 and 255"
        )


@dataclass(frozen=True, slots=True)
class PoseVisualizerConfig:
    """Small immutable configuration for deterministic OpenCV drawing.

    All colors are ordered as OpenCV BGR tuples.
    """

    keypoint_radius: int = 4
    keypoint_thickness: int = -1
    keypoint_color: Color = (0, 255, 0)
    skeleton_thickness: int = 2
    skeleton_color: Color = (0, 255, 255)
    bounding_box_thickness: int = 2
    bounding_box_color: Color = (255, 0, 0)
    draw_keypoints: bool = True
    draw_skeleton: bool = True
    draw_bounding_boxes: bool = False
    minimum_keypoint_confidence: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "keypoint_radius",
            "skeleton_thickness",
            "bounding_box_thickness",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise InvalidVisualizationConfigError(
                    f"{field_name} must be a positive integer"
                )

        if (
            isinstance(self.keypoint_thickness, bool)
            or not isinstance(self.keypoint_thickness, int)
            or self.keypoint_thickness == 0
            or self.keypoint_thickness < -1
        ):
            raise InvalidVisualizationConfigError(
                "keypoint_thickness must be -1 (filled) or a positive integer"
            )

        for field_name in (
            "keypoint_color",
            "skeleton_color",
            "bounding_box_color",
        ):
            _validate_color(getattr(self, field_name), field_name)

        for field_name in (
            "draw_keypoints",
            "draw_skeleton",
            "draw_bounding_boxes",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise InvalidVisualizationConfigError(
                    f"{field_name} must be a boolean"
                )

        threshold = self.minimum_keypoint_confidence
        if threshold is not None:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, Real)
                or not isfinite(threshold)
                or not 0.0 <= threshold <= 1.0
            ):
                raise InvalidVisualizationConfigError(
                    "minimum_keypoint_confidence must be between 0 and 1"
                )


def normalized_to_pixel(
    x: float, y: float, image_width: int, image_height: int
) -> Pixel | None:
    """Convert an in-frame normalized point into a drawable pixel.

    Coordinates outside ``[0, 1]`` or non-finite values return ``None``.
    Conversion uses ``floor(normalized * dimension)`` and caps the exact 1.0
    boundary at the final valid pixel, so `(1, 1)` maps to
    ``(image_width - 1, image_height - 1)``.
    """
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, int)
        or image_width <= 0
        or isinstance(image_height, bool)
        or not isinstance(image_height, int)
        or image_height <= 0
    ):
        raise InvalidVisualizationInputError(
            "image dimensions must be positive integers"
        )
    try:
        finite = isfinite(x) and isfinite(y)
    except TypeError:
        return None
    if not finite or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return (
        min(int(x * image_width), image_width - 1),
        min(int(y * image_height), image_height - 1),
    )


def _validate_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise InvalidVisualizationInputError("image must be a NumPy array")
    if image.size == 0:
        raise InvalidVisualizationInputError("image must not be empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidVisualizationInputError(
            "image must have shape (height, width, 3)"
        )
    if image.dtype != np.uint8:
        raise InvalidVisualizationInputError("image dtype must be uint8")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise InvalidVisualizationInputError("image dimensions must be positive")
    return image


def _keypoint_pixel(
    keypoint: Keypoint,
    image_width: int,
    image_height: int,
    minimum_confidence: float | None,
) -> Pixel | None:
    confidence = keypoint.confidence
    if confidence is not None:
        try:
            if not isfinite(confidence):
                return None
        except TypeError:
            return None
        if minimum_confidence is not None and confidence < minimum_confidence:
            return None
    return normalized_to_pixel(keypoint.x, keypoint.y, image_width, image_height)


def _bounding_box_pixels(
    bbox: BoundingBox, image_width: int, image_height: int
) -> tuple[Pixel, Pixel] | None:
    values = (bbox.x, bbox.y, bbox.width, bbox.height)
    try:
        if not all(isfinite(value) for value in values):
            return None
    except TypeError:
        return None
    if bbox.width < 0.0 or bbox.height < 0.0:
        return None

    left = bbox.x
    top = bbox.y
    right = bbox.x + bbox.width
    bottom = bbox.y + bbox.height
    if right < 0.0 or bottom < 0.0 or left > 1.0 or top > 1.0:
        return None

    clipped_left = min(max(left, 0.0), 1.0)
    clipped_top = min(max(top, 0.0), 1.0)
    clipped_right = min(max(right, 0.0), 1.0)
    clipped_bottom = min(max(bottom, 0.0), 1.0)
    start = normalized_to_pixel(
        clipped_left, clipped_top, image_width, image_height
    )
    end = normalized_to_pixel(
        clipped_right, clipped_bottom, image_width, image_height
    )
    if start is None or end is None:
        return None
    return start, end


class PoseVisualizer:
    """Render unified pose data onto a copy of an OpenCV BGR image."""

    def __init__(self, config: PoseVisualizerConfig | None = None) -> None:
        if config is not None and not isinstance(config, PoseVisualizerConfig):
            raise InvalidVisualizationConfigError(
                "config must be a PoseVisualizerConfig instance"
            )
        self.config = config or PoseVisualizerConfig()

    def render(self, image: object, pose_result: PoseResult) -> np.ndarray:
        source = _validate_image(image)
        if not isinstance(pose_result, PoseResult):
            raise InvalidVisualizationInputError(
                "pose_result must be a unified PoseResult"
            )

        rendered = source.copy()
        image_height, image_width = rendered.shape[:2]
        try:
            for person in pose_result.people:
                keypoints = {point.name: point for point in person.keypoints}
                pixels: dict[str, Pixel] = {}
                for name, point in keypoints.items():
                    pixel = _keypoint_pixel(
                        point,
                        image_width,
                        image_height,
                        self.config.minimum_keypoint_confidence,
                    )
                    if pixel is not None:
                        pixels[name] = pixel

                if self.config.draw_bounding_boxes and person.bbox is not None:
                    box = _bounding_box_pixels(
                        person.bbox, image_width, image_height
                    )
                    if box is not None:
                        cv2.rectangle(
                            rendered,
                            box[0],
                            box[1],
                            self.config.bounding_box_color,
                            self.config.bounding_box_thickness,
                            lineType=cv2.LINE_8,
                        )

                if self.config.draw_skeleton:
                    for start_name, end_name in CANONICAL_SKELETON:
                        start = pixels.get(start_name)
                        end = pixels.get(end_name)
                        if start is not None and end is not None:
                            cv2.line(
                                rendered,
                                start,
                                end,
                                self.config.skeleton_color,
                                self.config.skeleton_thickness,
                                lineType=cv2.LINE_8,
                            )

                if self.config.draw_keypoints:
                    for point in person.keypoints:
                        pixel = pixels.get(point.name)
                        if pixel is not None:
                            cv2.circle(
                                rendered,
                                pixel,
                                self.config.keypoint_radius,
                                self.config.keypoint_color,
                                self.config.keypoint_thickness,
                                lineType=cv2.LINE_8,
                            )
        except cv2.error as error:
            raise VisualizationRenderingError(
                f"OpenCV failed while rendering pose data: {error}"
            ) from error
        return rendered
