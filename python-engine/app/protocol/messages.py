"""Protocol-v1 request parsing, response envelopes, and error mapping."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ..core import BackendUnavailableError, UnsupportedBackendError
from ..models import (
    BackendInferenceError,
    BackendInitializationError,
    InvalidConfigurationError,
    InvalidInputImageError,
    ModelAssetNotFoundError,
)


PROTOCOL_VERSION = 1
SUPPORTED_COMMANDS = frozenset(
    {"ping", "get_backends", "estimate", "estimate_frame", "shutdown"}
)
RequestId = str | int
JsonObject = dict[str, Any]


def _reject_non_finite_json(token: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {token}")


@dataclass(frozen=True, slots=True)
class ProtocolRequest:
    """One validated protocol-v1 request object."""

    request_id: RequestId
    request_type: str
    fields: JsonObject


class ProtocolError(Exception):
    """A predictable request failure with a stable wire error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        request_id: RequestId | None = None,
    ) -> None:
        self.code = code
        self.request_id = request_id
        super().__init__(message)


def parse_request(line: str) -> ProtocolRequest:
    """Parse and validate one complete NDJSON request line."""
    try:
        value = json.loads(line, parse_constant=_reject_non_finite_json)
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ProtocolError(
            "invalid_json",
            "Request line is not valid JSON",
        ) from error

    if not isinstance(value, dict):
        raise ProtocolError(
            "invalid_request",
            "Request JSON must be an object",
        )

    request_id = value.get("id")
    if (
        "id" not in value
        or isinstance(request_id, bool)
        or not isinstance(request_id, (str, int))
    ):
        raise ProtocolError(
            "invalid_request",
            "Request id must be a string or integer",
        )

    version = value.get("protocol_version", PROTOCOL_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ProtocolError(
            "unsupported_protocol_version",
            f"Unsupported protocol version: {version!r}; expected 1",
            request_id=request_id,
        )

    request_type = value.get("type")
    if not isinstance(request_type, str) or not request_type.strip():
        raise ProtocolError(
            "invalid_request",
            "Request type must be a non-empty string",
            request_id=request_id,
        )
    request_type = request_type.strip()
    if request_type not in SUPPORTED_COMMANDS:
        raise ProtocolError(
            "unsupported_command",
            f"Unsupported command: {request_type!r}",
            request_id=request_id,
        )

    return ProtocolRequest(request_id, request_type, value)


def success_response(request_id: RequestId, result: JsonObject) -> JsonObject:
    """Build the single protocol-v1 success envelope."""
    return {"id": request_id, "ok": True, "result": result}


def error_response(error: ProtocolError) -> JsonObject:
    """Build the single protocol-v1 failure envelope."""
    return {
        "id": error.request_id,
        "ok": False,
        "error": {"code": error.code, "message": str(error)},
    }


def map_exception(
    error: Exception,
    *,
    request_id: RequestId | None,
) -> ProtocolError:
    """Map project exceptions to stable protocol-v1 error categories."""
    if isinstance(error, ProtocolError):
        if error.request_id is not None or request_id is None:
            return error
        return ProtocolError(error.code, str(error), request_id=request_id)
    if isinstance(error, UnsupportedBackendError):
        code = "unsupported_backend"
    elif isinstance(error, BackendUnavailableError):
        code = "backend_unavailable"
    elif isinstance(error, ModelAssetNotFoundError):
        code = "model_asset_not_found"
    elif isinstance(error, InvalidConfigurationError):
        code = "invalid_backend_configuration"
    elif isinstance(error, BackendInitializationError):
        code = "backend_initialization_failed"
    elif isinstance(error, (BackendInferenceError, InvalidInputImageError)):
        code = "backend_inference_failed"
    else:
        code = "internal_error"
        error = RuntimeError("An unexpected internal error occurred")
    return ProtocolError(code, str(error), request_id=request_id)


def encode_response(response: JsonObject) -> str:
    """Encode one compact strict-JSON response line."""
    return json.dumps(
        response,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
