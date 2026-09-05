"""Typed pose-estimation errors shared across backend implementations."""

from __future__ import annotations

from pathlib import Path


class PoseEstimationError(Exception):
    """Base class for predictable pose-estimation failures."""


class InvalidInputImageError(PoseEstimationError):
    """The supplied decoded image does not satisfy the estimator contract."""


class ModelAssetNotFoundError(PoseEstimationError):
    """A required backend model asset does not exist at the resolved path."""

    def __init__(self, expected_path: Path) -> None:
        self.expected_path = expected_path
        super().__init__(
            "MediaPipe Pose Landmarker model asset is missing. "
            f"Expected a file at: {expected_path}. "
            "Pass model_path in MediaPipePoseConfig to configure another location."
        )


class InvalidConfigurationError(PoseEstimationError):
    """Backend configuration contains an unsupported or invalid value."""


class BackendInitializationError(PoseEstimationError):
    """A backend could not initialize its reusable inference resources."""


class BackendInferenceError(PoseEstimationError):
    """A backend failed while performing or normalizing inference."""


class EstimatorClosedError(PoseEstimationError):
    """Inference was requested after the estimator had been closed."""


class BackendResourceError(PoseEstimationError):
    """A backend resource could not be released cleanly."""
