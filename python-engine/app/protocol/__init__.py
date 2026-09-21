"""Public API for the local sidecar protocol."""

from .messages import (
    PROTOCOL_VERSION,
    ProtocolError,
    ProtocolRequest,
    encode_response,
    parse_request,
)
from .serialization import PoseSerializationError, serialize_pose_result
from .sidecar import PoseSidecar

__all__ = [
    "PROTOCOL_VERSION",
    "PoseSerializationError",
    "PoseSidecar",
    "ProtocolError",
    "ProtocolRequest",
    "encode_response",
    "parse_request",
    "serialize_pose_result",
]
