"""Persistent sequential NDJSON sidecar for the unified pose engine."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Callable, Protocol, TextIO

import cv2
import numpy as np

from ..core import PoseBackend, PoseEngine, PoseEngineConfig
from ..models import PoseResult
from ..models.base import ImageArray
from .messages import (
    PROTOCOL_VERSION,
    JsonObject,
    ProtocolError,
    ProtocolRequest,
    RequestId,
    encode_response,
    error_response,
    map_exception,
    parse_request,
    success_response,
)
from .serialization import serialize_pose_result


class Engine(Protocol):
    """The PoseEngine surface owned by the sidecar."""

    def estimate(self, image: ImageArray) -> PoseResult: ...

    def close(self) -> None: ...


EngineFactory = Callable[[PoseEngineConfig], Engine]
ImageLoader = Callable[[str, int], np.ndarray | None]

@dataclass(frozen=True, slots=True)
class _BackendSelection:
    backend: PoseBackend
    model_path: str | None
    engine_config: PoseEngineConfig

    @property
    def signature(self) -> tuple[PoseBackend, str | None]:
        return (self.backend, self.model_path)


class PoseSidecar:
    """Own protocol streams and lazily cached backend engines."""

    def __init__(
        self,
        stdin: TextIO,
        stdout: TextIO,
        stderr: TextIO,
        *,
        engine_factory: EngineFactory = PoseEngine,
        image_loader: ImageLoader = cv2.imread,
    ) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self._engine_factory = engine_factory
        self._image_loader = image_loader
        self._engines: dict[PoseBackend, Engine] = {}
        self._engine_signatures: dict[
            PoseBackend, tuple[PoseBackend, str | None]
        ] = {}

    @property
    def initialized_backends(self) -> tuple[PoseBackend, ...]:
        """Expose deterministic cache state for diagnostics and tests."""
        return tuple(self._engines)

    def run(self) -> int:
        """Process requests in order until shutdown or normal stdin EOF."""
        try:
            for line in self.stdin:
                response, should_stop = self.process_line(line)
                self._write_response(response)
                if should_stop:
                    break
        except Exception:
            self._write_diagnostic("Fatal sidecar request-loop failure")
            traceback.print_exc(file=self.stderr)
            return 1
        finally:
            self.close()
        return 0

    def process_line(self, line: str) -> tuple[JsonObject, bool]:
        """Handle one request line without allowing recoverable errors to escape."""
        request_id: RequestId | None = None
        try:
            request = parse_request(line)
            request_id = request.request_id
            return self._dispatch(request)
        except Exception as error:
            mapped = map_exception(error, request_id=request_id)
            if mapped.code == "internal_error":
                self._write_diagnostic("Unexpected request handling failure")
                traceback.print_exception(error, file=self.stderr)
            return error_response(mapped), False

    def _dispatch(self, request: ProtocolRequest) -> tuple[JsonObject, bool]:
        if request.request_type == "ping":
            result = {"status": "ready", "protocol_version": PROTOCOL_VERSION}
            return success_response(request.request_id, result), False
        if request.request_type == "get_backends":
            result = {
                "available": [
                    backend.value for backend in PoseEngine.available_backends()
                ],
                "unavailable": [PoseBackend.MMPOSE.value],
            }
            return success_response(request.request_id, result), False
        if request.request_type == "estimate":
            result = self._estimate(request)
            return success_response(request.request_id, result), False
        if request.request_type == "shutdown":
            self.close()
            result = {"status": "shutdown"}
            return success_response(request.request_id, result), True
        raise AssertionError(f"Unvalidated command: {request.request_type!r}")

    def _estimate(self, request: ProtocolRequest) -> JsonObject:
        selection = self._parse_backend_selection(request)
        image = self._load_image(request)
        engine = self._get_engine(selection, request.request_id)
        with redirect_stdout(self.stderr):
            result = engine.estimate(image)
        return serialize_pose_result(result)

    def _parse_backend_selection(
        self,
        request: ProtocolRequest,
    ) -> _BackendSelection:
        backend_value = request.fields.get("backend")
        if not isinstance(backend_value, str) or not backend_value.strip():
            raise ProtocolError(
                "invalid_request",
                "backend must be a non-empty string",
                request_id=request.request_id,
            )
        base_config = PoseEngineConfig(
            backend=backend_value
        )
        backend = base_config.backend
        if not isinstance(backend, PoseBackend):
            raise AssertionError("PoseEngineConfig did not normalize its backend")

        raw_config = request.fields.get("backend_config", {})
        if not isinstance(raw_config, dict):
            raise ProtocolError(
                "invalid_backend_configuration",
                "backend_config must be an object",
                request_id=request.request_id,
            )
        unexpected = set(raw_config) - {"model_path"}
        if unexpected:
            fields = ", ".join(sorted(str(item) for item in unexpected))
            raise ProtocolError(
                "invalid_backend_configuration",
                f"Unsupported backend_config field(s): {fields}",
                request_id=request.request_id,
            )

        raw_model_path = raw_config.get("model_path")
        if raw_model_path is not None and not isinstance(raw_model_path, str):
            raise ProtocolError(
                "invalid_backend_configuration",
                "backend_config.model_path must be a string or null",
                request_id=request.request_id,
            )
        model_path = self._normalize_model_path(raw_model_path, request.request_id)

        if backend is PoseBackend.MEDIAPIPE:
            with redirect_stdout(self.stderr):
                from ..models.mediapipe_pose import MediaPipePoseConfig

                backend_config = MediaPipePoseConfig(model_path=model_path)
            engine_config = PoseEngineConfig(
                backend=backend,
                mediapipe=backend_config,
            )
        elif backend is PoseBackend.YOLO:
            with redirect_stdout(self.stderr):
                from ..models.yolo_pose import YoloPoseConfig

                backend_config = YoloPoseConfig(model_path=model_path)
            engine_config = PoseEngineConfig(backend=backend, yolo=backend_config)
        else:
            engine_config = base_config
        return _BackendSelection(backend, model_path, engine_config)

    @staticmethod
    def _normalize_model_path(
        value: str | None,
        request_id: RequestId,
    ) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ProtocolError(
                "invalid_backend_configuration",
                "backend_config.model_path must not be empty",
                request_id=request_id,
            )
        try:
            return str(Path(value).expanduser().resolve())
        except (OSError, RuntimeError) as error:
            raise ProtocolError(
                "invalid_backend_configuration",
                "backend_config.model_path could not be resolved",
                request_id=request_id,
            ) from error

    def _get_engine(
        self,
        selection: _BackendSelection,
        request_id: RequestId,
    ) -> Engine:
        existing = self._engines.get(selection.backend)
        if existing is not None:
            if self._engine_signatures[selection.backend] != selection.signature:
                raise ProtocolError(
                    "invalid_backend_configuration",
                    f"Backend {selection.backend.value!r} is already initialized "
                    "with a different configuration",
                    request_id=request_id,
                )
            return existing

        with redirect_stdout(self.stderr):
            engine = self._engine_factory(selection.engine_config)
        self._engines[selection.backend] = engine
        self._engine_signatures[selection.backend] = selection.signature
        return engine

    def _load_image(self, request: ProtocolRequest) -> ImageArray:
        value = request.fields.get("image_path")
        if not isinstance(value, str) or not value.strip():
            raise ProtocolError(
                "invalid_request",
                "image_path must be a non-empty local path string",
                request_id=request.request_id,
            )
        if value.lower().startswith(("http://", "https://")):
            raise ProtocolError(
                "invalid_request",
                "image_path must refer to a local file, not a URL",
                request_id=request.request_id,
            )
        try:
            path = Path(value).expanduser().resolve()
        except (OSError, RuntimeError) as error:
            raise ProtocolError(
                "invalid_request",
                "image_path could not be resolved",
                request_id=request.request_id,
            ) from error
        if not path.is_file():
            raise ProtocolError(
                "image_not_found",
                f"Image file does not exist or is not a regular file: {path}",
                request_id=request.request_id,
            )
        try:
            with redirect_stdout(self.stderr):
                image = self._image_loader(str(path), cv2.IMREAD_COLOR)
        except (cv2.error, OSError, RuntimeError, ValueError) as error:
            raise ProtocolError(
                "image_decode_failed",
                f"Image could not be decoded: {path}",
                request_id=request.request_id,
            ) from error
        if (
            not isinstance(image, np.ndarray)
            or image.size == 0
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise ProtocolError(
                "image_decode_failed",
                f"Image could not be decoded as an OpenCV BGR uint8 image: {path}",
                request_id=request.request_id,
            )
        return image

    def close(self) -> None:
        """Close every cached engine once, continuing after cleanup failures."""
        engines = tuple(self._engines.items())
        self._engines.clear()
        self._engine_signatures.clear()
        for backend, engine in engines:
            try:
                with redirect_stdout(self.stderr):
                    engine.close()
            except Exception:
                self._write_diagnostic(
                    f"Failed to close {backend.value!r} pose engine"
                )
                traceback.print_exc(file=self.stderr)

    def _write_response(self, response: JsonObject) -> None:
        self.stdout.write(encode_response(response) + "\n")
        self.stdout.flush()

    def _write_diagnostic(self, message: str) -> None:
        self.stderr.write(message + "\n")
        self.stderr.flush()
