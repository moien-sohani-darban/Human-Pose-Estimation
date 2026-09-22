"""Task 13 tests for development and frozen resource-root resolution."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from app import runtime
from app.models import assets


def test_development_resource_root_is_package_based_and_cwd_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = Path(runtime.__file__).resolve().parent.parent
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.chdir(tmp_path)

    assert runtime.get_runtime_resource_root() == expected


def test_frozen_resource_root_uses_pyinstaller_internal_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frozen_root = tmp_path / "installed app" / "python-sidecar" / "_internal"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(frozen_root), raising=False)

    assert runtime.get_runtime_resource_root() == frozen_root.resolve()


def test_logical_model_paths_do_not_change_for_packaging() -> None:
    assert assets.MEDIAPIPE_MODEL.display_path == (
        "models/mediapipe/pose_landmarker.task"
    )
    assert assets.YOLO_MODEL.display_path == "models/yolo/yolo11n-pose.pt"
