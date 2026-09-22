"""Central runtime and resource-root detection for development and packaging."""

from __future__ import annotations

from pathlib import Path
import sys


def get_runtime_resource_root() -> Path:
    """Return the immutable root containing application model resources.

    Development resolves from the checked-out ``python-engine`` package. A
    PyInstaller build resolves from its private ``_internal`` directory via
    ``sys._MEIPASS``. No path depends on the process working directory.
    """
    frozen_root = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and frozen_root:
        return Path(frozen_root).resolve()
    return Path(__file__).resolve().parent.parent
