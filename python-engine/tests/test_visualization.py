"""Deterministic tests for backend-independent pose visualization."""

from copy import deepcopy
from math import nan

import numpy as np
import pytest

from app.models import BoundingBox, Keypoint, PersonPose, PoseResult
from app.utils import (
    CANONICAL_SKELETON,
    InvalidVisualizationConfigError,
    InvalidVisualizationInputError,
    PoseVisualizer,
    PoseVisualizerConfig,
    normalized_to_pixel,
)
from app.utils.visualization import CANONICAL_KEYPOINT_NAMES


def make_keypoint(
    index: int,
    name: str,
    x: float,
    y: float,
    confidence: float | None = 1.0,
) -> Keypoint:
    return Keypoint(index, name, x, y, 0.0, confidence)


def make_result(*people: PersonPose) -> PoseResult:
    return PoseResult(True, "synthetic", list(people))


@pytest.mark.parametrize(
    ("x", "y", "width", "height", "expected"),
    [
        (0.0, 0.0, 10, 20, (0, 0)),
        (1.0, 1.0, 10, 20, (9, 19)),
        (0.5, 0.5, 10, 20, (5, 10)),
        (0.5, 0.5, 1, 1, (0, 0)),
        (-0.1, 0.5, 10, 20, None),
        (0.5, 1.1, 10, 20, None),
        (nan, 0.5, 10, 20, None),
        (0.5, float("inf"), 10, 20, None),
    ],
)
def test_normalized_to_pixel_conversion(
    x: float,
    y: float,
    width: int,
    height: int,
    expected: tuple[int, int] | None,
) -> None:
    assert normalized_to_pixel(x, y, width, height) == expected


@pytest.mark.parametrize(("width", "height"), [(0, 1), (1, 0), (-1, 2), (True, 2)])
def test_normalized_to_pixel_rejects_invalid_dimensions(
    width: int, height: int
) -> None:
    with pytest.raises(InvalidVisualizationInputError):
        normalized_to_pixel(0.5, 0.5, width, height)


@pytest.mark.parametrize(
    "image",
    [
        None,
        "not an image",
        np.array([], dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 1), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_render_rejects_invalid_images(image: object) -> None:
    with pytest.raises(InvalidVisualizationInputError):
        PoseVisualizer().render(image, make_result())


def test_render_rejects_non_unified_result() -> None:
    with pytest.raises(InvalidVisualizationInputError, match="PoseResult"):
        PoseVisualizer().render(
            np.zeros((4, 4, 3), dtype=np.uint8), object()  # type: ignore[arg-type]
        )


def test_empty_result_returns_unchanged_copy() -> None:
    image = np.full((8, 9, 3), 17, dtype=np.uint8)
    original = image.copy()

    rendered = PoseVisualizer().render(image, make_result())

    assert rendered is not image
    assert np.array_equal(rendered, original)
    assert np.array_equal(image, original)


def test_single_person_draws_keypoints_and_skeleton() -> None:
    image = np.zeros((21, 21, 3), dtype=np.uint8)
    person = PersonPose(
        0,
        [
            make_keypoint(11, "left_shoulder", 0.25, 0.5),
            make_keypoint(12, "right_shoulder", 0.75, 0.5),
        ],
        None,
    )
    config = PoseVisualizerConfig(
        keypoint_radius=1,
        keypoint_color=(0, 255, 0),
        skeleton_thickness=1,
        skeleton_color=(0, 0, 255),
    )

    rendered = PoseVisualizer(config).render(image, make_result(person))

    assert tuple(rendered[10, 5]) == (0, 255, 0)
    assert tuple(rendered[10, 15]) == (0, 255, 0)
    assert tuple(rendered[10, 10]) == (0, 0, 255)
    assert rendered.shape == image.shape
    assert rendered.dtype == image.dtype


def test_multiple_people_are_all_rendered() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    people = (
        PersonPose(7, [make_keypoint(0, "nose", 0.25, 0.25)], None),
        PersonPose(12, [make_keypoint(0, "nose", 0.75, 0.75)], None),
    )
    config = PoseVisualizerConfig(
        keypoint_radius=1,
        draw_skeleton=False,
        keypoint_color=(10, 20, 30),
    )

    rendered = PoseVisualizer(config).render(image, make_result(*people))

    assert tuple(rendered[5, 5]) == (10, 20, 30)
    assert tuple(rendered[15, 15]) == (10, 20, 30)
    assert [person.person_id for person in people] == [7, 12]


def test_off_frame_non_finite_and_low_confidence_points_are_skipped() -> None:
    points = [
        make_keypoint(0, "nose", -0.5, 0.5),
        make_keypoint(1, "left_eye_inner", 0.5, 0.5),
        make_keypoint(2, "left_eye", 0.75, 0.75, confidence=0.2),
        make_keypoint(3, "left_eye_outer", 0.25, 0.25, confidence=None),
    ]
    object.__setattr__(points[1], "x", nan)
    config = PoseVisualizerConfig(
        keypoint_radius=1,
        draw_skeleton=False,
        minimum_keypoint_confidence=0.5,
    )

    rendered = PoseVisualizer(config).render(
        np.zeros((20, 20, 3), dtype=np.uint8),
        make_result(PersonPose(0, points, None)),
    )

    assert not np.any(rendered[10, 0])
    assert not np.any(rendered[15, 15])
    assert tuple(rendered[5, 5]) == config.keypoint_color


def test_canonical_skeleton_is_valid_and_covers_major_topology() -> None:
    undirected_edges = {frozenset(edge) for edge in CANONICAL_SKELETON}

    assert len(undirected_edges) == len(CANONICAL_SKELETON)
    assert all(start != end for start, end in CANONICAL_SKELETON)
    assert all(
        start in CANONICAL_KEYPOINT_NAMES and end in CANONICAL_KEYPOINT_NAMES
        for start, end in CANONICAL_SKELETON
    )
    for expected in (
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_heel"),
        ("right_heel", "right_foot_index"),
    ):
        assert frozenset(expected) in undirected_edges


def test_bounding_box_draws_and_preserves_contract_data() -> None:
    bbox = BoundingBox(0.25, 0.25, 0.5, 0.5)
    person = PersonPose(0, [], bbox)
    result = make_result(person)
    before = deepcopy(result)
    config = PoseVisualizerConfig(
        draw_keypoints=False,
        draw_skeleton=False,
        draw_bounding_boxes=True,
        bounding_box_thickness=1,
        bounding_box_color=(255, 10, 20),
    )

    rendered = PoseVisualizer(config).render(
        np.zeros((20, 20, 3), dtype=np.uint8), result
    )

    assert tuple(rendered[5, 5]) == config.bounding_box_color
    assert tuple(rendered[15, 15]) == config.bounding_box_color
    assert result == before


def test_partially_off_frame_box_is_clipped_for_rendering_only() -> None:
    bbox = BoundingBox(0.0, 0.25, 0.25, 0.5)
    object.__setattr__(bbox, "x", -0.25)
    original_values = (bbox.x, bbox.y, bbox.width, bbox.height)
    config = PoseVisualizerConfig(
        draw_keypoints=False,
        draw_skeleton=False,
        draw_bounding_boxes=True,
        bounding_box_thickness=1,
    )

    rendered = PoseVisualizer(config).render(
        np.zeros((20, 20, 3), dtype=np.uint8),
        make_result(PersonPose(0, [], bbox)),
    )

    assert tuple(rendered[5, 0]) == config.bounding_box_color
    assert (bbox.x, bbox.y, bbox.width, bbox.height) == original_values


def test_missing_bounding_box_is_skipped() -> None:
    config = PoseVisualizerConfig(
        draw_keypoints=False,
        draw_skeleton=False,
        draw_bounding_boxes=True,
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    rendered = PoseVisualizer(config).render(
        image, make_result(PersonPose(0, [], None))
    )

    assert np.array_equal(rendered, image)


def test_render_does_not_modify_pose_result() -> None:
    result = make_result(
        PersonPose(
            4,
            [
                make_keypoint(11, "left_shoulder", 0.2, 0.3),
                make_keypoint(12, "right_shoulder", 0.8, 0.3),
            ],
            BoundingBox(0.2, 0.3, 0.6, 0.5),
        )
    )
    before = deepcopy(result)

    PoseVisualizer(
        PoseVisualizerConfig(draw_bounding_boxes=True)
    ).render(np.zeros((20, 20, 3), dtype=np.uint8), result)

    assert result == before


def test_rendering_is_deterministic() -> None:
    person = PersonPose(0, [make_keypoint(0, "nose", 0.5, 0.5)], None)
    image = np.zeros((15, 15, 3), dtype=np.uint8)
    visualizer = PoseVisualizer()

    first = visualizer.render(image, make_result(person))
    second = visualizer.render(image, make_result(person))

    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"keypoint_radius": 0},
        {"keypoint_thickness": 0},
        {"keypoint_thickness": -2},
        {"skeleton_thickness": 0},
        {"bounding_box_thickness": -1},
        {"keypoint_color": (0, 255)},
        {"skeleton_color": (0, 0, 256)},
        {"bounding_box_color": (0, True, 0)},
        {"draw_keypoints": 1},
        {"minimum_keypoint_confidence": -0.1},
        {"minimum_keypoint_confidence": 1.1},
        {"minimum_keypoint_confidence": nan},
    ],
)
def test_invalid_visualizer_configuration_is_rejected(
    kwargs: dict[str, object]
) -> None:
    with pytest.raises(InvalidVisualizationConfigError):
        PoseVisualizerConfig(**kwargs)  # type: ignore[arg-type]


def test_visualizer_rejects_wrong_configuration_type() -> None:
    with pytest.raises(InvalidVisualizationConfigError):
        PoseVisualizer(object())  # type: ignore[arg-type]
