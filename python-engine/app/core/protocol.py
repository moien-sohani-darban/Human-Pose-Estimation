"""Public identifiers and configuration for the unified pose engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..models.errors import PoseEstimationError

if TYPE_CHECKING:
    from ..models.mediapipe_pose import MediaPipePoseConfig
    from ..models.yolo_pose import YoloPoseConfig


class PoseEngineError(PoseEstimationError):
    """Base class for pose-engine orchestration failures."""


class UnsupportedBackendError(PoseEngineError):
    """The requested backend identifier is not recognized."""


class BackendUnavailableError(PoseEngineError):
    """A known backend cannot run in the current project environment."""


class InvalidBackendResultError(PoseEngineError):
    """A selected estimator returned something other than ``PoseResult``."""


class PoseBackend(str, Enum):
    """Stable identifiers for known pose-estimation backends."""

    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"
    MMPOSE = "mmpose"


AVAILABLE_BACKENDS: tuple[PoseBackend, ...] = (
    PoseBackend.MEDIAPIPE,
    PoseBackend.YOLO,
)


@dataclass(frozen=True, slots=True)
class PoseEngineConfig:
    """Select one backend and optionally provide its native configuration."""

    backend: PoseBackend | str = PoseBackend.MEDIAPIPE
    mediapipe: MediaPipePoseConfig | None = None
    yolo: YoloPoseConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.backend, PoseBackend):
            return
        if not isinstance(self.backend, str):
            raise UnsupportedBackendError(
                "backend must be 'mediapipe', 'yolo', or 'mmpose'"
            )
        try:
            backend = PoseBackend(self.backend.strip().lower())
        except ValueError as error:
            raise UnsupportedBackendError(
                f"Unsupported pose backend: {self.backend!r}"
            ) from error
        object.__setattr__(self, "backend", backend)
