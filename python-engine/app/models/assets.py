"""Canonical local model assets and lightweight availability metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from ..runtime import get_runtime_resource_root


ENGINE_ROOT: Path = get_runtime_resource_root()


@dataclass(frozen=True, slots=True)
class ModelAssetDefinition:
    """Stable identity and project-relative location for one model asset."""

    backend: str
    model_name: str
    relative_path: Path

    @property
    def display_path(self) -> str:
        return self.relative_path.as_posix()


@dataclass(frozen=True, slots=True)
class ModelAssetStatus:
    """Filesystem presence metadata; presence does not prove loadability."""

    backend: str
    model_name: str
    default_path: str
    display_path: str
    exists: bool
    size_bytes: int | None

    def to_dict(self) -> dict[str, str | bool | int | None]:
        """Return protocol-safe primitive values."""
        return asdict(self)


MEDIAPIPE_MODEL: Final = ModelAssetDefinition(
    backend="mediapipe",
    model_name="mediapipe-pose-landmarker",
    relative_path=Path("models/mediapipe/pose_landmarker.task"),
)
YOLO_MODEL: Final = ModelAssetDefinition(
    backend="yolo",
    model_name="yolo11n-pose",
    relative_path=Path("models/yolo/yolo11n-pose.pt"),
)
MODEL_ASSETS: Final = {
    MEDIAPIPE_MODEL.backend: MEDIAPIPE_MODEL,
    YOLO_MODEL.backend: YOLO_MODEL,
}


def _backend_value(backend: object) -> str:
    value = getattr(backend, "value", backend)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported model backend: {backend!r}")
    return value.strip().lower()


def get_model_asset_definition(backend: object) -> ModelAssetDefinition:
    """Return the canonical asset definition for a supported backend."""
    backend_id = _backend_value(backend)
    try:
        return MODEL_ASSETS[backend_id]
    except KeyError as error:
        raise ValueError(f"Unsupported model backend: {backend_id!r}") from error


def get_default_model_path(backend: object) -> Path:
    """Resolve a canonical path from this package, never from process cwd."""
    definition = get_model_asset_definition(backend)
    return (ENGINE_ROOT / definition.relative_path).resolve()


def inspect_default_model_asset(backend: object) -> ModelAssetStatus:
    """Inspect one canonical file without importing or initializing a backend."""
    definition = get_model_asset_definition(backend)
    absolute_path = get_default_model_path(definition.backend)
    try:
        stat = absolute_path.stat()
    except FileNotFoundError:
        exists = False
        size_bytes = None
    else:
        exists = absolute_path.is_file()
        size_bytes = stat.st_size if exists else None
    return ModelAssetStatus(
        backend=definition.backend,
        model_name=definition.model_name,
        default_path=definition.display_path,
        display_path=definition.display_path,
        exists=exists,
        size_bytes=size_bytes,
    )


def inspect_default_model_assets() -> dict[str, ModelAssetStatus]:
    """Inspect all supported defaults in stable definition order."""
    return {
        backend: inspect_default_model_asset(backend)
        for backend in MODEL_ASSETS
    }
