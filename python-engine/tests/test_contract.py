"""Tests for the backend-independent pose data contract."""

import pytest

from app.models import BoundingBox, Keypoint, PersonPose, PoseResult


def test_pose_contract_preserves_normalized_values_and_metadata() -> None:
    keypoint = Keypoint(0, "nose", 0.25, 0.75, -0.12, 0.9)
    bbox = BoundingBox(0.25, 0.5, 0.5, 0.4)
    person = PersonPose(0, (keypoint,), bbox)
    result = PoseResult(True, "test", (person,), 640, 480, 1.25)

    assert result.success is True
    assert result.people[0].keypoints[0] == keypoint
    assert result.people[0].bbox == bbox
    assert (result.image_width, result.image_height) == (640, 480)


def test_out_of_frame_keypoints_are_preserved_without_clamping() -> None:
    keypoint = Keypoint(0, "nose", -0.1, 1.2, None, None)

    assert keypoint.x == -0.1
    assert keypoint.y == 1.2


def test_empty_result_has_no_people() -> None:
    result = PoseResult(True, "mediapipe", ())

    assert result.people == []


def test_person_and_keypoint_indices_must_be_deterministic() -> None:
    keypoint = Keypoint(0, "nose", 0.5, 0.5, 0.0, 1.0)

    with pytest.raises(ValueError, match="keypoint indices must be unique"):
        PersonPose(0, (keypoint, keypoint), None)
    with pytest.raises(ValueError, match="person IDs must be unique"):
        PoseResult(
            True,
            "test",
            (PersonPose(0, (), None), PersonPose(0, (), None)),
        )


def test_bounding_box_must_fit_normalized_image() -> None:
    with pytest.raises(ValueError, match="fit within normalized image bounds"):
        BoundingBox(0.8, 0.2, 0.3, 0.4)
