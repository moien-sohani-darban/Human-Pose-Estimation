const ERROR_MESSAGES = {
  model_asset_not_found:
    "This backend needs a compatible local model. Select one with Browse and try again.",
  unsupported_backend:
    "This pose backend is not supported. Choose MediaPipe or YOLO Pose.",
  backend_unavailable:
    "This pose backend is unavailable in the current environment. Choose another backend.",
  sidecar_timeout:
    "Pose estimation took too long, so the local engine was reset. Try again.",
  sidecar_spawn_failed:
    "The local pose engine could not start. Restart the application and try again.",
  sidecar_runtime_missing:
    "The packaged pose engine is unavailable. Reinstall the application and try again.",
  sidecar_not_running:
    "The local pose engine stopped unexpectedly. Restart the application and try again.",
  sidecar_exited:
    "The local pose engine stopped unexpectedly. Try the action again to restart it.",
  sidecar_read_failed:
    "The application could not read the pose-engine response. Restart the application and try again.",
  sidecar_write_failed:
    "The application could not send work to the pose engine. Restart the application and try again.",
  sidecar_state_failed:
    "The local pose engine entered an invalid state. Restart the application and try again.",
  sidecar_task_failed:
    "The local pose task stopped unexpectedly. Restart the application and try again.",
  protocol_serialization_failed:
    "The pose request could not be prepared. Restart the application and try again.",
  protocol_parse_failed:
    "The pose engine returned an invalid response. Restart the application and try again.",
  protocol_id_mismatch:
    "The pose-engine response could not be matched to this request. Restart the application and try again.",
  invalid_backend_configuration:
    "The selected backend configuration cannot be used. If you changed its model, restart the application and try again.",
  invalid_request:
    "The pose request was invalid. Re-select the input and try again.",
  image_not_found:
    "The selected image could not be found. Choose the image again.",
  image_decode_failed:
    "The selected file could not be decoded as an image. Choose a supported image file.",
  backend_initialization_failed:
    "The pose backend could not start. Select a compatible model and try again.",
  backend_inference_failed:
    "The pose backend could not process this input. Check the input and model, then try again.",
  invalid_frame_data:
    "The captured media frame was invalid. Stop and restart pose analysis, then try again.",
  frame_decode_failed:
    "The captured media frame could not be decoded. Stop and restart pose analysis.",
  frame_encode_failed:
    "The media frame could not be prepared. Stop and restart pose analysis.",
  frame_too_large:
    "The captured media frame was too large to process safely. Reduce the input resolution and try again.",
  internal_error:
    "The local pose engine encountered an unexpected problem. Restart the application and try again.",
};

const GENERIC_ERROR_MESSAGE =
  "Something unexpected happened. Check your input and try again.";

function parseError(error) {
  if (error && typeof error === "object") {
    return error;
  }

  if (typeof error === "string") {
    try {
      const parsed = JSON.parse(error);
      if (parsed && typeof parsed === "object") {
        return parsed;
      }
    } catch {
      if (!error.includes("\n") && !error.toLowerCase().includes("traceback")) {
        return { message: error };
      }
    }
  }

  return {};
}

export function normalizeCommandError(error) {
  const parsed = parseError(error);
  const code =
    typeof parsed.code === "string"
      ? parsed.code
      : typeof parsed.kind === "string"
        ? parsed.kind
        : "unexpected_error";

  return {
    code,
    message: ERROR_MESSAGES[code] ?? GENERIC_ERROR_MESSAGE,
  };
}

export function buildEstimateArguments({ backend, imagePath, modelPath }) {
  const cleanModelPath =
    typeof modelPath === "string" && modelPath.trim()
      ? modelPath.trim()
      : null;

  return {
    request: {
      backend,
      imagePath,
      modelPath: cleanModelPath,
    },
  };
}

export function buildEstimateFrameArguments({ backend, frameData, modelPath }) {
  const cleanModelPath =
    typeof modelPath === "string" && modelPath.trim()
      ? modelPath.trim()
      : null;

  return {
    request: {
      backend,
      frameData: Array.from(frameData),
      modelPath: cleanModelPath,
    },
  };
}
