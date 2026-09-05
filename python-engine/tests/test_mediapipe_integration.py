"""Optional integration test using a developer-provided model asset."""

from pathlib import Path

import cv2
import pytest

from app.models import PoseEstimator, PoseResult
from app.models.mediapipe_pose import DEFAULT_MODEL_PATH, MediaPipePoseEstimator


@pytest.mark.integration
def test_real_pose_landmarker_returns_unified_result() -> None:
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip(f"MediaPipe model asset is not installed at {DEFAULT_MODEL_PATH}")

    image = cv2.imread(str(Path(__file__).parent / "test-person.jfif"))
    assert image is not None

    with MediaPipePoseEstimator() as estimator:
        result = estimator.estimate(image)

    assert isinstance(estimator, PoseEstimator)
    assert isinstance(result, PoseResult)
    assert result.backend == "mediapipe"
    assert (result.image_width, result.image_height) == (365, 547)
