"""Optional offline integration test for a local YOLO pose model."""

from pathlib import Path

import cv2
import pytest

from app.models import PoseEstimator, PoseResult
from app.models.yolo_pose import DEFAULT_MODEL_PATH, YoloPoseEstimator


@pytest.mark.integration
def test_real_yolo_pose_model_returns_unified_result() -> None:
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip(f"YOLO pose model asset is not installed at {DEFAULT_MODEL_PATH}")

    image = cv2.imread(str(Path(__file__).parent / "test-person.jfif"))
    assert image is not None

    with YoloPoseEstimator() as estimator:
        result = estimator.estimate(image)

    assert isinstance(estimator, PoseEstimator)
    assert isinstance(result, PoseResult)
    assert result.backend == "yolo"
    assert (result.image_width, result.image_height) == (365, 547)
    assert all(len(person.keypoints) == 17 for person in result.people)
    assert all(point.z is None for person in result.people for point in person.keypoints)
