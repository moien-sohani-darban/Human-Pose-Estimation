"""PyInstaller bootstrap that preserves the canonical sidecar application."""

from __future__ import annotations

from contextlib import redirect_stdout
import os
import sys

from app.main import main


def verify_packaged_imports() -> None:
    """Load the complete supported runtime stack for packaging smoke tests."""
    with redirect_stdout(sys.stderr):
        import cv2
        import mediapipe
        import numpy
        import torch
        import torchvision
        import ultralytics
        import app

    versions = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "mediapipe": mediapipe.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "app": app.__name__,
    }
    print(
        "[packaged-import-check] "
        + " ".join(f"{name}={version}" for name, version in versions.items()),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    if os.environ.get("HPE_PACKAGED_IMPORT_CHECK") == "1":
        verify_packaged_imports()
    raise SystemExit(main())
