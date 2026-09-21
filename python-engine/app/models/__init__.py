"""Public backend-neutral pose model API."""

from .base import BoundingBox, Keypoint, PersonPose, PoseEstimator, PoseResult
from .assets import (
    ModelAssetStatus,
    get_default_model_path,
    inspect_default_model_asset,
    inspect_default_model_assets,
)
from .errors import (
    BackendInferenceError,
    BackendInitializationError,
    BackendResourceError,
    EstimatorClosedError,
    InvalidConfigurationError,
    InvalidInputImageError,
    ModelAssetNotFoundError,
    PoseEstimationError,
)

__all__ = [
    "BackendInferenceError",
    "BackendInitializationError",
    "BackendResourceError",
    "BoundingBox",
    "EstimatorClosedError",
    "InvalidConfigurationError",
    "InvalidInputImageError",
    "Keypoint",
    "ModelAssetNotFoundError",
    "ModelAssetStatus",
    "PersonPose",
    "PoseEstimationError",
    "PoseEstimator",
    "PoseResult",
    "get_default_model_path",
    "inspect_default_model_asset",
    "inspect_default_model_assets",
]
