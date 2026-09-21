"""Canonical executable entry point for the persistent Python sidecar."""

from __future__ import annotations

import sys

from .protocol import PoseSidecar


def _configure_standard_streams() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main() -> int:
    """Run protocol v1 until a shutdown request or normal stdin EOF."""
    _configure_standard_streams()
    return PoseSidecar(sys.stdin, sys.stdout, sys.stderr).run()


if __name__ == "__main__":
    raise SystemExit(main())
