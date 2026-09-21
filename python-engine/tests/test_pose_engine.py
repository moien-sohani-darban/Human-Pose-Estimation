"""Deterministic tests for unified pose-engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from app.core import (
    AVAILABLE_BACKENDS,
    BackendUnavailableError,
    InvalidBackendResultError,
    PoseBackend,
    PoseEngine,
    PoseEngineConfig,
    UnsupportedBackendError,
)
from app.core import pose_engine
from app.models import (
    BackendInferenceError,
    BackendInitializationError,
    EstimatorClosedError,
    InvalidConfigurationError,
    PoseEstimator,
    PoseResult,
)


@dataclass
class FakeEstimator(PoseEstimator):
    result: object
    backend_name: str = "fake"
    estimate_calls: list[object] | None = None
    close_calls: int = 0

    def __post_init__(self) -> None:
        self.estimate_calls = []

    @property
    def backend(self) -> str:
        return self.backend_name

    def estimate(self, image: object) -> object:
        assert self.estimate_calls is not None
        self.estimate_calls.append(image)
        return self.result

    def close(self) -> None:
        self.close_calls += 1


def patch_backend_helpers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mediapipe: object,
    yolo: object,
) -> None:
    monkeypatch.setattr(pose_engine, "_create_mediapipe_estimator", mediapipe)
    monkeypatch.setattr(pose_engine, "_create_yolo_estimator", yolo)


def test_default_config_selects_only_mediapipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = PoseResult(True, "mediapipe", [])
    estimator = FakeEstimator(result, "mediapipe")
    calls: list[object] = []

    def create_mediapipe(config: object) -> FakeEstimator:
        calls.append(config)
        return estimator

    def reject_yolo(config: object) -> FakeEstimator:
        raise AssertionError("YOLO must not initialize")

    patch_backend_helpers(
        monkeypatch, mediapipe=create_mediapipe, yolo=reject_yolo
    )

    engine = PoseEngine()

    assert engine.backend is PoseBackend.MEDIAPIPE
    assert calls == [None]


def test_yolo_selection_initializes_only_yolo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeEstimator(PoseResult(True, "yolo", []), "yolo")
    calls: list[object] = []

    def reject_mediapipe(config: object) -> FakeEstimator:
        raise AssertionError("MediaPipe must not initialize")

    def create_yolo(config: object) -> FakeEstimator:
        calls.append(config)
        return estimator

    patch_backend_helpers(
        monkeypatch, mediapipe=reject_mediapipe, yolo=create_yolo
    )

    engine = PoseEngine(PoseEngineConfig(backend="YOLO"))

    assert engine.backend is PoseBackend.YOLO
    assert calls == [None]


@pytest.mark.parametrize(
    ("backend", "config_field"),
    [
        (PoseBackend.MEDIAPIPE, "mediapipe"),
        (PoseBackend.YOLO, "yolo"),
    ],
)
def test_backend_specific_config_is_forwarded_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    backend: PoseBackend,
    config_field: str,
) -> None:
    backend_config = object()
    estimator = FakeEstimator(PoseResult(True, backend.value, []), backend.value)
    received: list[object] = []

    def create(config: object) -> FakeEstimator:
        received.append(config)
        return estimator

    patch_backend_helpers(monkeypatch, mediapipe=create, yolo=create)
    kwargs = {"backend": backend, config_field: backend_config}

    PoseEngine(PoseEngineConfig(**kwargs))  # type: ignore[arg-type]

    assert received == [backend_config]


def test_estimate_delegates_exact_image_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = PoseResult(True, "yolo", [])
    estimator = FakeEstimator(result)
    monkeypatch.setattr(pose_engine, "_create_estimator", lambda config: estimator)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    engine = PoseEngine(PoseEngineConfig(PoseBackend.YOLO))

    returned = engine.estimate(image)

    assert returned is result
    assert estimator.estimate_calls == [image]


def test_non_pose_result_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    estimator = FakeEstimator(object())
    monkeypatch.setattr(pose_engine, "_create_estimator", lambda config: estimator)
    engine = PoseEngine()

    with pytest.raises(InvalidBackendResultError):
        engine.estimate(np.zeros((1, 1, 3), dtype=np.uint8))


def test_inference_error_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = BackendInferenceError("inference failed")
    estimator = FakeEstimator(PoseResult(True, "mediapipe", []))

    def fail(image: object) -> object:
        raise failure

    estimator.estimate = fail  # type: ignore[method-assign]
    monkeypatch.setattr(pose_engine, "_create_estimator", lambda config: estimator)
    engine = PoseEngine()

    with pytest.raises(BackendInferenceError) as raised:
        engine.estimate(np.zeros((1, 1, 3), dtype=np.uint8))

    assert raised.value is failure


def test_repeated_estimates_reuse_one_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeEstimator(PoseResult(True, "mediapipe", []))
    construction_calls = 0

    def create(config: PoseEngineConfig) -> FakeEstimator:
        nonlocal construction_calls
        construction_calls += 1
        return estimator

    monkeypatch.setattr(pose_engine, "_create_estimator", create)
    engine = PoseEngine()
    image = np.zeros((1, 1, 3), dtype=np.uint8)

    engine.estimate(image)
    engine.estimate(image)

    assert construction_calls == 1
    assert estimator.estimate_calls == [image, image]


def test_close_is_idempotent_and_blocks_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeEstimator(PoseResult(True, "mediapipe", []))
    monkeypatch.setattr(pose_engine, "_create_estimator", lambda config: estimator)
    engine = PoseEngine()
    engine.close()
    engine.close()

    with pytest.raises(EstimatorClosedError):
        engine.estimate(np.zeros((1, 1, 3), dtype=np.uint8))

    assert estimator.close_calls == 1
    assert estimator.estimate_calls == []


@pytest.mark.parametrize("raise_inside", [False, True])
def test_context_manager_always_closes(
    monkeypatch: pytest.MonkeyPatch,
    raise_inside: bool,
) -> None:
    estimator = FakeEstimator(PoseResult(True, "mediapipe", []))
    monkeypatch.setattr(pose_engine, "_create_estimator", lambda config: estimator)

    if raise_inside:
        with pytest.raises(RuntimeError, match="inside"):
            with PoseEngine():
                raise RuntimeError("inside")
    else:
        with PoseEngine() as engine:
            assert engine.backend is PoseBackend.MEDIAPIPE

    assert estimator.close_calls == 1


def test_mmpose_is_known_but_unavailable_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(config: object) -> FakeEstimator:
        raise AssertionError("No available backend may initialize")

    patch_backend_helpers(monkeypatch, mediapipe=reject, yolo=reject)

    with pytest.raises(BackendUnavailableError, match="Task 05"):
        PoseEngine(PoseEngineConfig(PoseBackend.MMPOSE))


@pytest.mark.parametrize("backend", ["banana", "", 42, None])
def test_unknown_backend_is_typed(backend: object) -> None:
    with pytest.raises(UnsupportedBackendError):
        PoseEngineConfig(backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("selected", "other_helper"),
    [
        (PoseBackend.MEDIAPIPE, "_create_yolo_estimator"),
        (PoseBackend.YOLO, "_create_mediapipe_estimator"),
    ],
)
def test_initialization_failure_preserves_error_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    selected: PoseBackend,
    other_helper: str,
) -> None:
    failure = BackendInitializationError(f"{selected.value} failed")

    def fail(config: object) -> FakeEstimator:
        raise failure

    def reject(config: object) -> FakeEstimator:
        raise AssertionError("Fallback backend must not initialize")

    monkeypatch.setattr(
        pose_engine,
        f"_create_{selected.value}_estimator",
        fail,
    )
    monkeypatch.setattr(pose_engine, other_helper, reject)

    with pytest.raises(BackendInitializationError) as raised:
        PoseEngine(PoseEngineConfig(selected))

    assert raised.value is failure


def test_available_backends_excludes_mmpose() -> None:
    assert AVAILABLE_BACKENDS == (PoseBackend.MEDIAPIPE, PoseBackend.YOLO)
    assert PoseEngine.available_backends() is AVAILABLE_BACKENDS
    assert PoseBackend.MMPOSE not in PoseEngine.available_backends()


def test_engine_module_has_no_eager_backend_implementation_imports() -> None:
    module_globals = vars(pose_engine)

    assert "MediaPipePoseEstimator" not in module_globals
    assert "YoloPoseEstimator" not in module_globals
    assert not any(name.lower().startswith("mmpose") for name in module_globals)


def test_wrong_engine_config_type_is_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        PoseEngine(config=object())  # type: ignore[arg-type]
