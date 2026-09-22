import {
  createLiveScheduler,
  LIVE_INFERENCE_INTERVAL_MS,
} from "./livePose.js";

export const VIDEO_RESULT_STALE_THRESHOLD_SECONDS = 0.75;

export function formatVideoTime(value) {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const wholeSeconds = Math.floor(value);
  const minutes = Math.floor(wholeSeconds / 60);
  const seconds = String(wholeSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

export function createVideoSelection(current, path, createPreviewSource) {
  if (typeof path !== "string" || !path.trim()) return current;
  const cleanPath = path.trim();
  return {
    path: cleanPath,
    name: cleanPath.split(/[\\/]/).filter(Boolean).at(-1) ?? "Selected video",
    previewSource: createPreviewSource(cleanPath),
  };
}

export function createVideoAnalysisController({
  media,
  captureCurrentFrame,
  estimateFrame,
  onResult,
  onError,
  onInvalidate,
  onEnded,
  onTiming,
  staleThresholdSeconds = VIDEO_RESULT_STALE_THRESHOLD_SECONDS,
  intervalMs = LIVE_INFERENCE_INTERVAL_MS,
  schedule,
  cancel,
  now,
}) {
  let enabled = false;

  const canSample = () =>
    media.readyState >= 2 &&
    media.videoWidth > 0 &&
    media.videoHeight > 0 &&
    !media.paused &&
    !media.ended;

  const scheduler = createLiveScheduler({
    captureFrame: async () => {
      const capturedTime = media.currentTime;
      return {
        capturedTime,
        frameData: await captureCurrentFrame(),
      };
    },
    estimateFrame: async ({ capturedTime, frameData }) => ({
      capturedTime,
      result: await estimateFrame(frameData),
    }),
    onResult: ({ capturedTime, result }) => {
      const drift = Math.abs(media.currentTime - capturedTime);
      if (drift <= staleThresholdSeconds) onResult(result, capturedTime);
    },
    onError: (error) => {
      enabled = false;
      onError(error);
    },
    onTiming,
    intervalMs,
    ...(schedule ? { schedule } : {}),
    ...(cancel ? { cancel } : {}),
    ...(now ? { now } : {}),
  });

  return {
    start() {
      enabled = true;
      if (canSample()) scheduler.start();
    },
    stop() {
      enabled = false;
      scheduler.stop();
      onInvalidate();
    },
    handlePlay() {
      if (enabled && canSample()) scheduler.start();
    },
    handlePause() {
      scheduler.stop();
    },
    handleSeeking() {
      scheduler.stop();
      onInvalidate();
    },
    handleSeeked() {
      if (enabled && canSample()) scheduler.start();
    },
    handleEnded() {
      enabled = false;
      scheduler.stop();
      onEnded();
    },
    replaceSource() {
      enabled = false;
      scheduler.stop();
      onInvalidate();
    },
    isEnabled: () => enabled,
    isInFlight: scheduler.isInFlight,
  };
}
