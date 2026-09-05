"""Tests for the installable Python engine foundation."""

from __future__ import annotations

import importlib
import inspect
import sys

import pytest

import app
from app.models.base import PoseEstimator


@pytest.mark.parametrize("module_name", ("numpy", "cv2", "mediapipe", "pytest"))
def test_foundation_dependency_imports(module_name: str) -> None:
    """Every declared foundation or test dependency is importable."""
    assert importlib.import_module(module_name) is not None


def test_selected_python_version() -> None:
    """Task 01 remains on the selected Python runtime."""
    assert sys.version_info[:3] == (3, 13, 5)


def test_package_metadata_is_importable() -> None:
    """Editable installation exposes the engine package normally."""
    assert app.__version__ == "0.1.0"


def test_pose_estimator_remains_abstract() -> None:
    """Backend implementations continue to share an abstract boundary."""
    assert inspect.isabstract(PoseEstimator)

