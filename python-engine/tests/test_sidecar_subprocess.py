"""Subprocess verification for the canonical sidecar executable boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ENGINE_ROOT = Path(__file__).resolve().parents[1]


def run_process(request_lines: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "app.main"],
        cwd=ENGINE_ROOT,
        input="\n".join(request_lines) + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_real_subprocess_ping_and_clean_shutdown() -> None:
    completed = run_process(
        [
            '{"id":"ping","type":"ping"}',
            '{"id":"stop","type":"shutdown"}',
        ]
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]

    assert completed.returncode == 0
    assert responses == [
        {
            "id": "ping",
            "ok": True,
            "result": {"status": "ready", "protocol_version": 1},
        },
        {
            "id": "stop",
            "ok": True,
            "result": {"status": "shutdown"},
        },
    ]
    assert completed.stderr == ""


def test_real_subprocess_recovers_after_malformed_json() -> None:
    completed = run_process(
        [
            "{bad",
            '{"id":"ping","type":"ping"}',
            '{"id":"stop","type":"shutdown"}',
        ]
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]

    assert completed.returncode == 0
    assert len(responses) == 3
    assert responses[0]["id"] is None
    assert responses[0]["error"]["code"] == "invalid_json"
    assert responses[1]["id"] == "ping"
    assert responses[1]["ok"] is True
    assert responses[2]["result"]["status"] == "shutdown"
    assert completed.stderr == ""
