"""Deterministic unit tests for protocol v1 and the persistent sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app.core import (
    BackendUnavailableError,
    PoseBackend,
    PoseEngineConfig,
    UnsupportedBackendError,
)
from app.models import assets
from app.models import (
    BackendInferenceError,
    BackendInitializationError,
    BoundingBox,
    InvalidConfigurationError,
    Keypoint,
    ModelAssetNotFoundError,
    PersonPose,
    PoseResult,
)
from app.protocol import (
    PoseSerializationError,
    PoseSidecar,
    ProtocolError,
    encode_response,
    parse_request,
    serialize_pose_result,
)
from app.protocol.messages import map_exception


def pose_result(backend: str = "mediapipe", people: int = 1) -> PoseResult:
    poses = []
    for person_id in range(people):
        poses.append(
            PersonPose(
                person_id=person_id,
                keypoints=[
                    Keypoint(
                        index=0,
                        name="nose",
                        x=0.25 + person_id * 0.1,
                        y=0.5,
                        z=None if person_id else -0.1,
                        confidence=None if person_id else 0.9,
                    )
                ],
                bbox=(
                    None
                    if person_id
                    else BoundingBox(0.1, 0.2, 0.4, 0.6)
                ),
            )
        )
    return PoseResult(True, backend, poses, 20, 10, 2.5)


@dataclass
class FakeEngine:
    config: PoseEngineConfig
    result: PoseResult
    estimate_calls: list[np.ndarray]
    close_calls: int = 0

    def estimate(self, image: np.ndarray) -> PoseResult:
        self.estimate_calls.append(image)
        return self.result

    def close(self) -> None:
        self.close_calls += 1


class FakeEngineFactory:
    def __init__(self) -> None:
        self.engines: list[FakeEngine] = []

    def __call__(self, config: PoseEngineConfig) -> FakeEngine:
        backend = config.backend
        assert isinstance(backend, PoseBackend)
        if backend is PoseBackend.MMPOSE:
            raise BackendUnavailableError("MMPose is unavailable")
        engine = FakeEngine(config, pose_result(backend.value), [])
        self.engines.append(engine)
        return engine


def write_image(path: Path) -> Path:
    image = np.full((10, 20, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def estimate_request(
    request_id: str,
    backend: str,
    image_path: Path,
    *,
    model_path: Path | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "id": request_id,
        "type": "estimate",
        "backend": backend,
        "image_path": str(image_path),
    }
    if model_path is not None:
        request["backend_config"] = {"model_path": str(model_path)}
    return request


def estimate_frame_request(
    request_id: str,
    backend: str,
    frame_bytes: bytes,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "type": "estimate_frame",
        "backend": backend,
        "image_base64": base64.b64encode(frame_bytes).decode("ascii"),
    }

def run_sidecar(
    requests: list[str],
    *,
    factory: Any | None = None,
) -> tuple[int, list[dict[str, Any]], str, PoseSidecar]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    sidecar = PoseSidecar(
        io.StringIO("\n".join(requests) + "\n"),
        stdout,
        stderr,
        engine_factory=factory or FakeEngineFactory(),
    )
    exit_code = sidecar.run()
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return exit_code, responses, stderr.getvalue(), sidecar


def test_parse_valid_request_object() -> None:
    request = parse_request(
        '{"id":"abc","type":"ping","protocol_version":1}'
    )

    assert request.request_id == "abc"
    assert request.request_type == "ping"
    assert request.fields["protocol_version"] == 1


@pytest.mark.parametrize("request_id", ["request-1", 7, 0])
def test_request_id_accepts_strings_and_integers(request_id: object) -> None:
    request = parse_request(json.dumps({"id": request_id, "type": "ping"}))

    assert request.request_id == request_id


@pytest.mark.parametrize(
    "line",
    ["{bad", "", "not-json", '{"id":1,"type":"ping","value":NaN}'],
)
def test_malformed_json_is_typed(line: str) -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_request(line)

    assert raised.value.code == "invalid_json"
    assert raised.value.request_id is None


@pytest.mark.parametrize("value", [[], "hello", 123, None])
def test_non_object_json_is_invalid_request(value: object) -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_request(json.dumps(value))

    assert raised.value.code == "invalid_request"


def test_missing_command_type_preserves_valid_id() -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_request('{"id":"known"}')

    assert raised.value.code == "invalid_request"
    assert raised.value.request_id == "known"


def test_unsupported_command_is_distinct() -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_request('{"id":1,"type":"dance"}')

    assert raised.value.code == "unsupported_command"
    assert raised.value.request_id == 1


@pytest.mark.parametrize("version", [0, 2, "1", True, None])
def test_unsupported_protocol_version(version: object) -> None:
    line = json.dumps({"id": "v", "type": "ping", "protocol_version": version})

    with pytest.raises(ProtocolError) as raised:
        parse_request(line)

    assert raised.value.code == "unsupported_protocol_version"
    assert raised.value.request_id == "v"


@pytest.mark.parametrize("request_id", [None, True, {}, []])
def test_invalid_request_id_is_rejected(request_id: object) -> None:
    line = json.dumps({"id": request_id, "type": "ping"})

    with pytest.raises(ProtocolError) as raised:
        parse_request(line)

    assert raised.value.code == "invalid_request"
    assert raised.value.request_id is None


def test_missing_request_id_is_rejected() -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_request('{"type":"ping"}')

    assert raised.value.code == "invalid_request"


def test_ping_is_ready_without_initializing_an_engine() -> None:
    factory = FakeEngineFactory()
    code, responses, stderr, sidecar = run_sidecar(
        ['{"id":"ping-1","type":"ping"}'],
        factory=factory,
    )

    assert code == 0
    assert responses == [
        {
            "id": "ping-1",
            "ok": True,
            "result": {"status": "ready", "protocol_version": 1},
        }
    ]
    assert factory.engines == []
    assert sidecar.initialized_backends == ()
    assert stderr == ""


def test_get_backends_reports_actual_availability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(assets, "ENGINE_ROOT", tmp_path)

    code, responses, _, _ = run_sidecar(
        ['{"id":2,"type":"get_backends"}']
    )

    assert code == 0
    assert responses[0]["ok"] is True
    assert responses[0]["result"] == {
        "available": ["mediapipe", "yolo"],
        "unavailable": ["mmpose"],
        "models": {
            "mediapipe": {
                "backend": "mediapipe",
                "model_name": "mediapipe-pose-landmarker",
                "default_path": "models/mediapipe/pose_landmarker.task",
                "display_path": "models/mediapipe/pose_landmarker.task",
                "exists": False,
                "size_bytes": None,
            },
            "yolo": {
                "backend": "yolo",
                "model_name": "yolo11n-pose",
                "default_path": "models/yolo/yolo11n-pose.pt",
                "display_path": "models/yolo/yolo11n-pose.pt",
                "exists": False,
                "size_bytes": None,
            },
        },
    }


@pytest.mark.parametrize(
    ("mediapipe_present", "yolo_present"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_get_backends_model_metadata_tracks_files_without_initializing_engines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mediapipe_present: bool,
    yolo_present: bool,
) -> None:
    monkeypatch.setattr(assets, "ENGINE_ROOT", tmp_path)
    expected = {
        "mediapipe": mediapipe_present,
        "yolo": yolo_present,
    }

    for backend, present in expected.items():
        if present:
            path = assets.get_default_model_path(backend)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"asset")

    factory = FakeEngineFactory()

    _, responses, _, sidecar = run_sidecar(
        ['{"id":"assets","type":"get_backends"}'],
        factory=factory,
    )

    models = responses[0]["result"]["models"]

    assert {
        backend: metadata["exists"]
        for backend, metadata in models.items()
    } == expected

    assert factory.engines == []
    assert sidecar.initialized_backends == ()


def test_estimate_decodes_image_and_returns_unified_result(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")
    factory = FakeEngineFactory()
    request = json.dumps(estimate_request("e1", "mediapipe", image_path))

    code, responses, _, _ = run_sidecar([request], factory=factory)

    assert code == 0
    assert responses[0]["ok"] is True
    assert responses[0]["result"] == serialize_pose_result(
        factory.engines[0].result
    )
    decoded = factory.engines[0].estimate_calls[0]
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (10, 20, 3)
    assert decoded.dtype == np.uint8


def test_estimate_frame_decodes_memory_image_and_returns_unified_result() -> None:
    source = np.full((12, 18, 3), (10, 40, 220), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", source)
    assert encoded_ok
    factory = FakeEngineFactory()
    request = json.dumps(
        estimate_frame_request("frame-1", "mediapipe", encoded.tobytes())
    )

    code, responses, _, _ = run_sidecar([request], factory=factory)

    assert code == 0
    assert responses[0]["ok"] is True
    decoded = factory.engines[0].estimate_calls[0]
    assert decoded.shape == (12, 18, 3)
    assert decoded.dtype == np.uint8


def test_image_and_frame_requests_reuse_the_same_backend_engine(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")
    source = np.full((8, 9, 3), 80, dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", source)
    assert encoded_ok
    factory = FakeEngineFactory()
    requests = [
        json.dumps(estimate_request("path", "yolo", image_path)),
        json.dumps(estimate_frame_request("frame", "yolo", encoded.tobytes())),
    ]

    _, responses, _, _ = run_sidecar(requests, factory=factory)

    assert all(response["ok"] for response in responses)
    assert len(factory.engines) == 1
    assert len(factory.engines[0].estimate_calls) == 2


@pytest.mark.parametrize("value", [None, 42, "", "%%not-base64%%", "8J+YgA"])
def test_invalid_frame_data_is_typed_and_recoverable(value: object) -> None:
    request = {
        "id": "bad-frame",
        "type": "estimate_frame",
        "backend": "mediapipe",
        "image_base64": value,
    }
    _, responses, _, _ = run_sidecar(
        [json.dumps(request), '{"id":"after","type":"ping"}']
    )

    assert responses[0]["error"]["code"] == "invalid_frame_data"
    assert responses[1]["ok"] is True


def test_corrupt_encoded_frame_is_decode_failed() -> None:
    request = estimate_frame_request("corrupt", "mediapipe", b"not an image")

    _, responses, _, _ = run_sidecar([json.dumps(request)])

    assert responses[0]["error"]["code"] == "frame_decode_failed"


def test_frame_size_limit_is_checked_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.protocol.sidecar as sidecar_module

    monkeypatch.setattr(sidecar_module, "MAX_FRAME_BYTES", 3)
    monkeypatch.setattr(sidecar_module, "MAX_FRAME_BASE64_CHARS", 4)
    request = estimate_frame_request("large", "mediapipe", b"four")

    _, responses, _, _ = run_sidecar([json.dumps(request)])

    assert responses[0]["error"]["code"] == "frame_too_large"


def test_decoded_frame_size_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.protocol.sidecar as sidecar_module

    monkeypatch.setattr(sidecar_module, "MAX_FRAME_BYTES", 3)
    monkeypatch.setattr(sidecar_module, "MAX_FRAME_BASE64_CHARS", 100)
    request = estimate_frame_request("large-decoded", "mediapipe", b"four")

    _, responses, _, _ = run_sidecar([json.dumps(request)])

    assert responses[0]["error"]["code"] == "frame_too_large"


def test_empty_decoded_frame_is_invalid() -> None:
    request = estimate_frame_request("empty", "mediapipe", b"")

    _, responses, _, _ = run_sidecar([json.dumps(request)])

    assert responses[0]["error"]["code"] == "invalid_frame_data"

def test_same_backend_engine_is_created_once_and_reused(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")
    factory = FakeEngineFactory()
    request = json.dumps(estimate_request("e", "mediapipe", image_path))

    code, responses, _, _ = run_sidecar([request, request], factory=factory)

    assert code == 0
    assert len(responses) == 2
    assert len(factory.engines) == 1
    assert len(factory.engines[0].estimate_calls) == 2


def test_mediapipe_yolo_mediapipe_reuses_two_engines(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")
    factory = FakeEngineFactory()
    requests = [
        json.dumps(estimate_request("m1", "mediapipe", image_path)),
        json.dumps(estimate_request("y1", "yolo", image_path)),
        json.dumps(estimate_request("m2", "mediapipe", image_path)),
    ]

    code, responses, _, _ = run_sidecar(requests, factory=factory)

    assert code == 0
    assert [response["id"] for response in responses] == ["m1", "y1", "m2"]
    assert [engine.config.backend for engine in factory.engines] == [
        PoseBackend.MEDIAPIPE,
        PoseBackend.YOLO,
    ]
    assert [len(engine.estimate_calls) for engine in factory.engines] == [2, 1]


@pytest.mark.parametrize("backend", ["mediapipe", "yolo"])
def test_model_path_config_is_forwarded(
    tmp_path: Path,
    backend: str,
) -> None:
    image_path = write_image(tmp_path / "person.png")
    model_path = tmp_path / "models" / "pose.model"
    factory = FakeEngineFactory()
    request = json.dumps(
        estimate_request("config", backend, image_path, model_path=model_path)
    )

    _, responses, _, _ = run_sidecar([request], factory=factory)

    assert responses[0]["ok"] is True
    config = factory.engines[0].config
    backend_config = config.mediapipe if backend == "mediapipe" else config.yolo
    assert backend_config is not None
    assert backend_config.model_path == str(model_path.resolve())


def test_different_configuration_for_cached_backend_is_rejected(
    tmp_path: Path,
) -> None:
    image_path = write_image(tmp_path / "person.png")
    first_model = tmp_path / "first.model"
    second_model = tmp_path / "second.model"
    factory = FakeEngineFactory()
    requests = [
        json.dumps(
            estimate_request("first", "yolo", image_path, model_path=first_model)
        ),
        json.dumps(
            estimate_request("second", "yolo", image_path, model_path=second_model)
        ),
    ]

    _, responses, _, _ = run_sidecar(requests, factory=factory)

    assert responses[0]["ok"] is True
    assert responses[1]["error"]["code"] == "invalid_backend_configuration"
    assert len(factory.engines) == 1
    assert len(factory.engines[0].estimate_calls) == 1


@pytest.mark.parametrize(
    "backend_config",
    [[], {"model_path": 42}, {"device": "cpu"}, {"model_path": "  "}],
)
def test_invalid_backend_configuration_is_recoverable(
    tmp_path: Path,
    backend_config: object,
) -> None:
    image_path = write_image(tmp_path / "person.png")
    request = estimate_request("bad-config", "yolo", image_path)
    request["backend_config"] = backend_config

    _, responses, _, _ = run_sidecar([json.dumps(request)])

    assert responses[0]["error"]["code"] == "invalid_backend_configuration"


@pytest.mark.parametrize("image_path", [None, 42, "", "  "])
def test_invalid_image_path_field(image_path: object) -> None:
    request = {
        "id": "image",
        "type": "estimate",
        "backend": "mediapipe",
        "image_path": image_path,
    }

    _, responses, _, _ = run_sidecar([json.dumps(request)])

    assert responses[0]["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("backend", [None, 42, "", "  "])
def test_invalid_or_missing_backend_field(backend: object) -> None:
    request = {
        "id": "backend",
        "type": "estimate",
        "backend": backend,
        "image_path": "unused.png",
    }

    _, responses, _, _ = run_sidecar([json.dumps(request)])

    assert responses[0]["error"]["code"] == "invalid_request"


def test_missing_image_and_directory_are_image_not_found(tmp_path: Path) -> None:
    requests = [
        json.dumps(
            estimate_request("missing", "mediapipe", tmp_path / "missing.png")
        ),
        json.dumps(estimate_request("directory", "mediapipe", tmp_path)),
    ]

    _, responses, _, _ = run_sidecar(requests)

    assert [response["error"]["code"] for response in responses] == [
        "image_not_found",
        "image_not_found",
    ]


def test_corrupt_image_is_decode_failed(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    request = json.dumps(estimate_request("corrupt", "mediapipe", corrupt))

    _, responses, _, _ = run_sidecar([request])

    assert responses[0]["error"]["code"] == "image_decode_failed"


def test_url_input_is_explicitly_rejected() -> None:
    request = {
        "id": "url",
        "type": "estimate",
        "backend": "mediapipe",
        "image_path": "https://example.test/person.png",
    }

    _, responses, _, _ = run_sidecar([json.dumps(request)])

    assert responses[0]["error"]["code"] == "invalid_request"


def test_pose_result_serialization_is_complete_and_stable() -> None:
    serialized = serialize_pose_result(pose_result(people=2))

    assert serialized == {
        "success": True,
        "backend": "mediapipe",
        "people": [
            {
                "person_id": 0,
                "keypoints": [
                    {
                        "index": 0,
                        "name": "nose",
                        "x": 0.25,
                        "y": 0.5,
                        "z": -0.1,
                        "confidence": 0.9,
                    }
                ],
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.6},
            },
            {
                "person_id": 1,
                "keypoints": [
                    {
                        "index": 0,
                        "name": "nose",
                        "x": 0.35,
                        "y": 0.5,
                        "z": None,
                        "confidence": None,
                    }
                ],
                "bbox": None,
            },
        ],
        "image_width": 20,
        "image_height": 10,
        "processing_time_ms": 2.5,
    }


def test_empty_pose_result_serializes_as_successful_no_detection() -> None:
    result = PoseResult(True, "yolo", [], 20, 10, None)

    assert serialize_pose_result(result) == {
        "success": True,
        "backend": "yolo",
        "people": [],
        "image_width": 20,
        "image_height": 10,
        "processing_time_ms": None,
    }


def test_optional_non_finite_values_become_null_in_strict_json() -> None:
    result = pose_result()
    result.processing_time_ms = float("nan")
    keypoint = result.people[0].keypoints[0]
    object.__setattr__(keypoint, "z", float("inf"))
    object.__setattr__(keypoint, "confidence", float("-inf"))

    encoded = encode_response(
        {"id": "strict", "ok": True, "result": serialize_pose_result(result)}
    )

    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    decoded = json.loads(encoded)
    assert decoded["result"]["processing_time_ms"] is None
    assert decoded["result"]["people"][0]["keypoints"][0]["z"] is None
    assert decoded["result"]["people"][0]["keypoints"][0]["confidence"] is None


def test_non_finite_required_coordinate_is_rejected() -> None:
    result = pose_result()
    object.__setattr__(result.people[0].keypoints[0], "x", float("nan"))

    with pytest.raises(PoseSerializationError):
        serialize_pose_result(result)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (UnsupportedBackendError("unknown"), "unsupported_backend"),
        (BackendUnavailableError("unavailable"), "backend_unavailable"),
        (ModelAssetNotFoundError(Path("missing.task")), "model_asset_not_found"),
        (InvalidConfigurationError("invalid"), "invalid_backend_configuration"),
        (BackendInitializationError("initialization"), "backend_initialization_failed"),
        (BackendInferenceError("inference"), "backend_inference_failed"),
    ],
)
def test_project_errors_have_centralized_protocol_mapping(
    error: Exception,
    code: str,
) -> None:
    mapped = map_exception(error, request_id="mapping")

    assert mapped.code == code
    assert mapped.request_id == "mapping"


def test_unsupported_and_unavailable_backends_are_distinct(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")
    requests = [
        json.dumps(estimate_request("unknown", "other", image_path)),
        json.dumps(estimate_request("mmpose", "mmpose", image_path)),
    ]

    _, responses, _, _ = run_sidecar(requests)

    assert [response["error"]["code"] for response in responses] == [
        "unsupported_backend",
        "backend_unavailable",
    ]


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (ModelAssetNotFoundError(Path("missing.task")), "model_asset_not_found"),
        (InvalidConfigurationError("bad config"), "invalid_backend_configuration"),
        (BackendInitializationError("failed init"), "backend_initialization_failed"),
    ],
)
def test_engine_creation_failure_is_protocol_safe(
    tmp_path: Path,
    failure: Exception,
    expected_code: str,
) -> None:
    image_path = write_image(tmp_path / "person.png")

    def fail(config: PoseEngineConfig) -> FakeEngine:
        raise failure

    request = json.dumps(estimate_request("failure", "mediapipe", image_path))
    _, responses, _, _ = run_sidecar([request], factory=fail)

    assert responses[0]["error"]["code"] == expected_code


def test_backend_inference_failure_is_protocol_safe(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")

    class FailingEngine:
        def estimate(self, image: np.ndarray) -> PoseResult:
            raise BackendInferenceError("inference failed")

        def close(self) -> None:
            pass

    request = json.dumps(estimate_request("failure", "yolo", image_path))
    _, responses, _, _ = run_sidecar(
        [request], factory=lambda config: FailingEngine()
    )

    assert responses[0]["error"]["code"] == "backend_inference_failed"


def test_unexpected_error_is_safe_logged_and_recoverable(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")
    calls = 0

    def factory(config: PoseEngineConfig) -> FakeEngine:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("private details")
        return FakeEngine(config, pose_result(), [])

    requests = [
        json.dumps(estimate_request("failure", "mediapipe", image_path)),
        '{"id":"ping","type":"ping"}',
    ]
    _, responses, stderr, _ = run_sidecar(requests, factory=factory)

    assert responses[0]["error"] == {
        "code": "internal_error",
        "message": "An unexpected internal error occurred",
    }
    assert responses[1]["ok"] is True
    assert "private details" not in json.dumps(responses)
    assert "private details" in stderr


def test_malformed_json_then_ping_is_recoverable() -> None:
    _, responses, _, _ = run_sidecar(
        ["{bad", '{"id":"after","type":"ping"}']
    )

    assert responses[0]["id"] is None
    assert responses[0]["error"]["code"] == "invalid_json"
    assert responses[1]["id"] == "after"
    assert responses[1]["ok"] is True


def test_shutdown_closes_all_engines_once_and_stops_loop(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")
    factory = FakeEngineFactory()
    requests = [
        json.dumps(estimate_request("m", "mediapipe", image_path)),
        json.dumps(estimate_request("y", "yolo", image_path)),
        '{"id":"stop","type":"shutdown"}',
        '{"id":"ignored","type":"ping"}',
    ]

    code, responses, _, _ = run_sidecar(requests, factory=factory)

    assert code == 0
    assert [response["id"] for response in responses] == ["m", "y", "stop"]
    assert responses[-1] == {
        "id": "stop",
        "ok": True,
        "result": {"status": "shutdown"},
    }
    assert [engine.close_calls for engine in factory.engines] == [1, 1]


def test_eof_closes_initialized_engines_once(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "person.png")
    factory = FakeEngineFactory()
    request = json.dumps(estimate_request("eof", "mediapipe", image_path))

    code, _, _, _ = run_sidecar([request], factory=factory)

    assert code == 0
    assert factory.engines[0].close_calls == 1


def test_stdout_contains_only_json_and_backend_noise_goes_to_stderr(
    tmp_path: Path,
) -> None:
    image_path = write_image(tmp_path / "person.png")

    class NoisyEngine(FakeEngine):
        def estimate(self, image: np.ndarray) -> PoseResult:
            print("inference noise")
            return super().estimate(image)

        def close(self) -> None:
            print("close noise")
            super().close()

    def noisy_factory(config: PoseEngineConfig) -> NoisyEngine:
        print("initialization noise")
        return NoisyEngine(config, pose_result(), [])

    requests = [
        json.dumps(estimate_request("noise", "mediapipe", image_path)),
        '{"id":"stop","type":"shutdown"}',
    ]
    code, responses, stderr, _ = run_sidecar(requests, factory=noisy_factory)

    assert code == 0
    assert all(isinstance(response, dict) for response in responses)
    assert len(responses) == 2
    encoded_responses = json.dumps(responses)
    assert "initialization noise" not in encoded_responses
    assert "inference noise" not in encoded_responses
    assert "close noise" not in encoded_responses
    assert "initialization noise" in stderr
    assert "inference noise" in stderr
    assert "close noise" in stderr
