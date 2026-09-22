export const LIVE_CAPTURE_MAX_DIMENSION = 960;
export const LIVE_CAPTURE_JPEG_QUALITY = 0.75;
export const LIVE_INFERENCE_INTERVAL_MS = 200;
export const MAX_FRAME_BYTES = 4 * 1024 * 1024;

export function computeInferenceDimensions(
  width,
  height,
  maxDimension = LIVE_CAPTURE_MAX_DIMENSION,
) {
  if (
    ![width, height, maxDimension].every(Number.isFinite) ||
    width <= 0 ||
    height <= 0 ||
    maxDimension <= 0
  ) {
    throw new Error("Video dimensions must be positive finite numbers");
  }

  const scale = Math.min(1, maxDimension / Math.max(width, height));

  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

export async function requestCamera(mediaDevices) {
  if (!mediaDevices || typeof mediaDevices.getUserMedia !== "function") {
    throw {
      code: "camera_unavailable",
      message:
        "Camera access is unavailable. Check Windows camera permissions and connect a camera, then try again.",
    };
  }

  return mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });
}

export function stopMediaStream(stream) {
  if (!stream || typeof stream.getTracks !== "function") return;

  for (const track of stream.getTracks()) {
    if (track && typeof track.stop === "function") {
      track.stop();
    }
  }
}

export function normalizeCameraError(error) {
  if (
    typeof error?.code === "string" &&
    typeof error?.message === "string"
  ) {
    return { code: error.code, message: error.message };
  }

  const name = typeof error?.name === "string" ? error.name : "";

  if (name === "NotAllowedError" || name === "SecurityError") {
    return {
      code: "camera_permission_denied",
      message:
        "Camera permission was denied. Allow camera access and try again.",
    };
  }

  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return {
      code: "camera_not_found",
      message:
        "No usable camera was found. Connect or enable a camera, then try again.",
    };
  }

  if (name === "NotReadableError" || name === "TrackStartError") {
    return {
      code: "camera_in_use",
      message:
        "The camera could not start. Close other applications using it, then try again.",
    };
  }

  return {
    code: "camera_start_failed",
    message: "The camera could not be started. Check the device and try again.",
  };
}

export function canvasToJpegBytes(
  canvas,
  quality = LIVE_CAPTURE_JPEG_QUALITY,
) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          reject({
            code: "frame_encode_failed",
            message: "The captured camera frame could not be encoded.",
          });
          return;
        }

        if (blob.size > MAX_FRAME_BYTES) {
          reject({
            code: "frame_too_large",
            message:
              "The captured camera frame exceeds the 4 MiB safety limit.",
          });
          return;
        }

        resolve(new Uint8Array(await blob.arrayBuffer()));
      },
      "image/jpeg",
      quality,
    );
  });
}

export async function captureMediaFrame(
  media,
  canvas,
  context = null,
  {
    notReadyCode = "media_not_ready",
    notReadyMessage = "The media is not ready to capture a frame.",
  } = {},
) {
  if (
    !media ||
    !canvas ||
    media.readyState < 2 ||
    !Number.isFinite(media.videoWidth) ||
    !Number.isFinite(media.videoHeight) ||
    media.videoWidth <= 0 ||
    media.videoHeight <= 0
  ) {
    throw {
      code: notReadyCode,
      message: notReadyMessage,
    };
  }

  const dimensions = computeInferenceDimensions(
    media.videoWidth,
    media.videoHeight,
  );

  if (canvas.width !== dimensions.width) {
    canvas.width = dimensions.width;
  }

  if (canvas.height !== dimensions.height) {
    canvas.height = dimensions.height;
  }

  const drawingContext =
    context ?? canvas.getContext("2d", { alpha: false });

  if (!drawingContext) {
    throw {
      code: "frame_encode_failed",
      message: "The frame capture canvas is unavailable.",
    };
  }

  drawingContext.drawImage(
    media,
    0,
    0,
    dimensions.width,
    dimensions.height,
  );

  return {
    context: drawingContext,
    frameData: await canvasToJpegBytes(canvas),
  };
}

export function captureCameraFrame(video, canvas, context = null) {
  return captureMediaFrame(video, canvas, context, {
    notReadyCode: "camera_not_ready",
    notReadyMessage: "The camera is not ready to capture a frame.",
  });
}

export function createLiveScheduler({
  captureFrame,
  estimateFrame,
  onResult,
  onError,
  onTiming,
  intervalMs = LIVE_INFERENCE_INTERVAL_MS,
  schedule = setTimeout,
  cancel = clearTimeout,
  now = () => performance.now(),
}) {
  let active = false;
  let inFlight = false;
  let generation = 0;
  let timer = null;

  const clearTimer = () => {
    if (timer !== null) {
      cancel(timer);
    }
    timer = null;
  };

  const queue = (delay = intervalMs) => {
    if (!active || timer !== null || inFlight) return;

    const session = generation;
    timer = schedule(() => run(session), delay);
  };

  const run = async (session) => {
    timer = null;

    if (!active || session !== generation || inFlight) return;

    inFlight = true;
    const startedAt = now();

    try {
      const frameData = await captureFrame();

      if (!active || session !== generation) return;

      const result = await estimateFrame(frameData);

      if (!active || session !== generation) return;

      onResult(result);

      const completedAt = now();
      onTiming(completedAt - startedAt, completedAt);
    } catch (error) {
      if (active && session === generation) {
        active = false;
        generation += 1;
        clearTimer();
        onError(error);
      }
    } finally {
      inFlight = false;

      if (active) {
        queue(intervalMs);
      }
    }
  };

  return {
    start() {
      if (active) return;

      active = true;
      generation += 1;
      queue(0);
    },

    stop() {
      active = false;
      generation += 1;
      clearTimer();
    },

    isActive: () => active,
    isInFlight: () => inFlight,
  };
}
