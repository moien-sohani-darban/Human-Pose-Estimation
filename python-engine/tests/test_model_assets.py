"""Task 10 tests for canonical local model-asset behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models import ModelAssetNotFoundError
from app.models import assets
from app.models.mediapipe_pose import MediaPipePoseConfig, resolve_model_path
from app.models.yolo_pose import (
    YoloModelAssetNotFoundError,
    YoloPoseConfig,
    YoloPoseEstimator,
    resolve_yolo_model_path,
)


def test_canonical_model_paths_and_identities() -> None:
    mediapipe = assets.get_model_asset_definition("mediapipe")
    yolo = assets.get_model_asset_definition("yolo")

    assert mediapipe.model_name == "mediapipe-pose-landmarker"
    assert mediapipe.display_path == "models/mediapipe/pose_landmarker.task"
    assert yolo.model_name == "yolo11n-pose"
    assert yolo.display_path == "models/yolo/yolo11n-pose.pt"
    assert assets.get_default_model_path("mediapipe") == (
        assets.ENGINE_ROOT / mediapipe.relative_path
    ).resolve()
    assert assets.get_default_model_path("yolo") == (
        assets.ENGINE_ROOT / yolo.relative_path
    ).resolve()


def test_canonical_paths_are_independent_of_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    before = {
        backend: assets.get_default_model_path(backend)
        for backend in ("mediapipe", "yolo")
    }

    monkeypatch.chdir(tmp_path)

    assert {
        backend: assets.get_default_model_path(backend)
        for backend in ("mediapipe", "yolo")
    } == before


@pytest.mark.parametrize(
    ("backend", "resolver", "config_type", "error_type"),
    [
        ("mediapipe", resolve_model_path, MediaPipePoseConfig, ModelAssetNotFoundError),
        ("yolo", resolve_yolo_model_path, YoloPoseConfig, YoloModelAssetNotFoundError),
    ],
)
def test_missing_default_raises_typed_asset_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    resolver: object,
    config_type: object,
    error_type: type[Exception],
) -> None:
    monkeypatch.setattr(assets, "ENGINE_ROOT", tmp_path)

    with pytest.raises(error_type) as raised:
        resolver(config_type().model_path)  # type: ignore[operator]

    assert raised.value.expected_path == assets.get_default_model_path(backend)


@pytest.mark.parametrize(
    ("resolver", "filename"),
    [
        (resolve_model_path, "custom.task"),
        (resolve_yolo_model_path, "custom.pt"),
    ],
)
def test_explicit_path_overrides_missing_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resolver: object,
    filename: str,
) -> None:
    monkeypatch.setattr(assets, "ENGINE_ROOT", tmp_path / "missing-default-root")
    custom = tmp_path / filename
    custom.write_bytes(b"test fixture")

    assert resolver(custom) == custom.resolve()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("mediapipe_present", "yolo_present"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_asset_metadata_reports_each_presence_combination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mediapipe_present: bool,
    yolo_present: bool,
) -> None:
    monkeypatch.setattr(assets, "ENGINE_ROOT", tmp_path)
    expected = {"mediapipe": mediapipe_present, "yolo": yolo_present}
    for backend, present in expected.items():
        if present:
            path = assets.get_default_model_path(backend)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(backend.encode())

    statuses = assets.inspect_default_model_assets()

    assert list(statuses) == ["mediapipe", "yolo"]
    for backend, present in expected.items():
        status = statuses[backend]
        assert status.backend == backend
        assert status.exists is present
        assert status.size_bytes == (len(backend) if present else None)
        assert status.display_path.startswith(f"models/{backend}/")
        assert status.default_path == status.display_path


def test_missing_default_yolo_is_rejected_before_ultralytics_constructor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(assets, "ENGINE_ROOT", tmp_path)
    constructor_called = False

    def forbidden_constructor(*_args: object, **_kwargs: object) -> object:
        nonlocal constructor_called
        constructor_called = True
        raise AssertionError("Ultralytics constructor must not be reached")

    monkeypatch.setattr(
        "app.models.yolo_pose._UltralyticsYOLO",
        forbidden_constructor,
    )

    with pytest.raises(YoloModelAssetNotFoundError):
        YoloPoseEstimator()

    assert constructor_called is False
