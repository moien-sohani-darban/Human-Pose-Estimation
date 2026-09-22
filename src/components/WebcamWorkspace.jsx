import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { estimatePoseFrame } from "../api/pose";
import {
  captureCameraFrame,
  createLiveScheduler,
  normalizeCameraError,
  requestCamera,
  stopMediaStream,
} from "../utils/livePose";
import {
  backendLabel,
  formatPeopleCount,
  validatePoseResult,
  withModelSetupGuidance,
} from "../utils/pose";
import PoseOverlay from "./PoseOverlay";

const WebcamWorkspace = forwardRef(function WebcamWorkspace(
  {
    selectedBackend,
    modelPath,
    defaultModelPath,
    overlayOptions,
    onLiveChange,
    onPresentationChange,
  },
  ref,
) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const streamRef = useRef(null);
  const schedulerRef = useRef(null);
  const generationRef = useRef(0);
  const lastCompletionRef = useRef(null);

  const [cameraStatus, setCameraStatus] = useState("idle");
  const [cameraLabel, setCameraLabel] = useState("Camera");
  const [cameraError, setCameraError] = useState(null);
  const [liveStatus, setLiveStatus] = useState("idle");
  const [liveError, setLiveError] = useState(null);
  const [result, setResult] = useState(null);
  const [inferenceFps, setInferenceFps] = useState(null);

  const stopLive = useCallback(() => {
    schedulerRef.current?.stop();
    schedulerRef.current = null;
    setLiveStatus("idle");
    setInferenceFps(null);
    lastCompletionRef.current = null;
    onLiveChange?.(false);
  }, [onLiveChange]);

  const stopCamera = useCallback(() => {
    generationRef.current += 1;
    stopLive();
    stopMediaStream(streamRef.current);
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraStatus("idle");
    setCameraLabel("Camera");
    setCameraError(null);
    setLiveError(null);
    setResult(null);
  }, [stopLive]);

  useEffect(() => stopCamera, [stopCamera]);

  async function startCamera() {
    if (cameraStatus === "requesting" || cameraStatus === "active") return;

    const generation = generationRef.current + 1;
    generationRef.current = generation;

    setCameraStatus("requesting");
    setCameraError(null);

    try {
      const stream = await requestCamera(navigator.mediaDevices);

      if (generation !== generationRef.current) {
        stopMediaStream(stream);
        return;
      }

      streamRef.current = stream;

      const [track] = stream.getVideoTracks();
      setCameraLabel(track?.label || "Camera");

      for (const currentTrack of stream.getVideoTracks()) {
        currentTrack.addEventListener("ended", stopCamera, { once: true });
      }

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      if (generation === generationRef.current) {
        setCameraStatus("active");
      }
    } catch (error) {
      if (generation !== generationRef.current) return;

      stopMediaStream(streamRef.current);
      streamRef.current = null;

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }

      setCameraStatus("error");
      setCameraError(normalizeCameraError(error));
    }
  }

  async function captureFrame() {
    const captured = await captureCameraFrame(
      videoRef.current,
      canvasRef.current,
      contextRef.current,
    );

    contextRef.current = captured.context;
    return captured.frameData;
  }

  function startLive() {
    if (
      cameraStatus !== "active" ||
      !selectedBackend ||
      schedulerRef.current
    ) {
      return;
    }

    setLiveError(null);
    setResult(null);
    setLiveStatus("running");
    onLiveChange?.(true);

    const scheduler = createLiveScheduler({
      captureFrame,
      estimateFrame: (frameData) =>
        estimatePoseFrame({
          backend: selectedBackend,
          frameData,
          modelPath,
        }),
      onResult: (nextResult) => {
        if (!validatePoseResult(nextResult)) {
          throw {
            code: "unexpected_result",
            message: "The pose engine returned an unexpected live result.",
          };
        }

        setResult(nextResult);
      },
      onError: (error) => {
        schedulerRef.current = null;
        setLiveStatus("error");
        setLiveError(
          withModelSetupGuidance(
            error,
            selectedBackend,
            defaultModelPath,
          ),
        );
        onLiveChange?.(false);
      },
      onTiming: (durationMs, completedAt) => {
        const previous = lastCompletionRef.current;
        const elapsed =
          previous === null ? durationMs : completedAt - previous;

        lastCompletionRef.current = completedAt;
        setInferenceFps(elapsed > 0 ? 1000 / elapsed : null);
      },
    });

    schedulerRef.current = scheduler;
    scheduler.start();
  }

  useImperativeHandle(ref, () => ({
    startCamera,
    stopCamera,
    startLive,
    stopLive,
  }));

  useEffect(() => {
    onPresentationChange?.({
      cameraStatus,
      cameraLabel,
      liveStatus,
      result,
      inferenceFps,
      cameraError,
      liveError,
    });
  }, [
    cameraStatus,
    cameraLabel,
    liveStatus,
    result,
    inferenceFps,
    cameraError,
    liveError,
    onPresentationChange,
  ]);

  const cameraActive = cameraStatus === "active";
  const liveActive = liveStatus === "running";
  const peopleCount = result?.people?.length ?? 0;

  return (
    <section className="workspace webcam-workspace">
      <header className="workspace-heading">
        <div>
          <h2>Live camera</h2>
          <p>{cameraActive ? cameraLabel : "Camera is off"}</p>
        </div>

        {liveActive && (
          <span className="status-chip success">Live estimation active</span>
        )}
      </header>

      {!cameraActive ? (
        <div className="empty-stage">
          <div className="empty-icon" aria-hidden="true">?</div>
          <h2>Start the camera</h2>
          <p>
            Camera permission is requested only after you choose Start Camera.
          </p>

          <button
            type="button"
            className="primary-button compact-button"
            onClick={startCamera}
            disabled={cameraStatus === "requesting"}
          >
            {cameraStatus === "requesting"
              ? "Requesting camera..."
              : "Start Camera"}
          </button>
        </div>
      ) : (
        <>
          <div className="image-viewer webcam-viewer">
            <figure className="image-stage webcam-stage">
              <video
                ref={videoRef}
                muted
                playsInline
                aria-label="Live camera preview"
              />

              {result && (
                <PoseOverlay
                  result={result}
                  showSkeleton={overlayOptions.skeleton}
                  showKeypoints={overlayOptions.keypoints}
                  showBoxes={overlayOptions.boxes}
                />
              )}

              {result && (
                <div className="stage-badge">
                  {backendLabel(result.backend)} ?{" "}
                  {peopleCount === 0
                    ? "No people detected"
                    : `${formatPeopleCount(peopleCount)} detected`}
                  {Number.isFinite(inferenceFps)
                    ? ` ? ${inferenceFps.toFixed(1)} FPS`
                    : ""}
                </div>
              )}
            </figure>
          </div>

          <div className="webcam-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={stopCamera}
            >
              Stop Camera
            </button>

            <button
              type="button"
              className="primary-button"
              onClick={liveActive ? stopLive : startLive}
              disabled={!selectedBackend}
            >
              {liveActive ? "Stop Live Estimation" : "Start Live Estimation"}
            </button>
          </div>
        </>
      )}

      {cameraError && (
        <div className="workspace-error" role="alert">
          <strong>Camera error</strong>
          <p>{cameraError.message}</p>
        </div>
      )}

      {liveError && (
        <div className="workspace-error" role="alert">
          <strong>Pose analysis error</strong>
          <p>{liveError.message}</p>
        </div>
      )}

      {liveActive && result && peopleCount === 0 && (
        <div className="zero-result" role="status">
          No people detected. Live estimation is still running.
        </div>
      )}

      <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
    </section>
  );
});

export default WebcamWorkspace;
