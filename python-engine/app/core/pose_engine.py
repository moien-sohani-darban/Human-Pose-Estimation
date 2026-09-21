"""Backend-independent pose-estimator selection and lifecycle orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..models.base import ImageArray, PoseEstimator, PoseResult
from ..models.errors import EstimatorClosedError, InvalidConfigurationError
from .protocol import (
    AVAILABLE_BACKENDS,
    BackendUnavailableError,
    InvalidBackendResultError,
    PoseBackend,
    PoseEngineConfig,
)

if TYPE_CHECKING:
    from ..models.mediapipe_pose import MediaPipePoseConfig
    from ..models.yolo_pose import YoloPoseConfig


def _create_mediapipe_estimator(
    config: MediaPipePoseConfig | None,
) -> PoseEstimator:
    from ..models.mediapipe_pose import MediaPipePoseEstimator

    return MediaPipePoseEstimator(config)


def _create_yolo_estimator(config: YoloPoseConfig | None) -> PoseEstimator:
    from ..models.yolo_pose import YoloPoseEstimator

    return YoloPoseEstimator(config)


def _create_estimator(config: PoseEngineConfig) -> PoseEstimator:
    if config.backend is PoseBackend.MEDIAPIPE:
        return _create_mediapipe_estimator(config.mediapipe)
    if config.backend is PoseBackend.YOLO:
        return _create_yolo_estimator(config.yolo)
    if config.backend is PoseBackend.MMPOSE:
        raise BackendUnavailableError(
            "MMPose is currently unavailable because Task 05 established "
            "an OpenMMLab compatibility limitation for Python 3.13.5."
        )
    raise AssertionError(f"Unhandled known backend: {config.backend!r}")


class PoseEngine:
    """Own one selected estimator and expose a unified inference entry point."""

    def __init__(self, config: PoseEngineConfig | None = None) -> None:
        if config is not None and not isinstance(config, PoseEngineConfig):
            raise InvalidConfigurationError(
                "config must be a PoseEngineConfig instance"
            )
        self.config = config or PoseEngineConfig()
        self._estimator = _create_estimator(self.config)
        self._closed = False

    @property
    def backend(self) -> PoseBackend:
        """Return the backend fixed for this engine instance."""
        return cast(PoseBackend, self.config.backend)

    @staticmethod
    def available_backends() -> tuple[PoseBackend, ...]:
        """Return the deterministic set of currently usable backends."""
        return AVAILABLE_BACKENDS

    def estimate(self, image: ImageArray) -> PoseResult:
        """Delegate one decoded image to the selected estimator unchanged."""
        if self._closed:
            raise EstimatorClosedError("Pose engine has already been closed")
        result = self._estimator.estimate(image)
        if not isinstance(result, PoseResult):
            raise InvalidBackendResultError(
                "Selected estimator returned a non-PoseResult value"
            )
        return result

    def close(self) -> None:
        """Release the owned estimator exactly once."""
        if self._closed:
            return
        self._closed = True
        self._estimator.close()

    def __enter__(self) -> PoseEngine:
        return self

    def __exit__(
        self, exc_type: object, exc_value: object, traceback: object
    ) -> None:
        self.close()
