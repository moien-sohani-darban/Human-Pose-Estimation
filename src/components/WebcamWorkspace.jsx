import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { estimatePoseFrame } from "../api/pose";
import {
  captureMediaFrame,
  createLiveScheduler,
  normalizeCameraError,
  requestCamera,
  stopMediaStream,
} from "../utils/livePose";
import { backendLabel, formatPeopleCount, validatePoseResult, withModelSetupGuidance } from "../utils/pose";
import Icon from "./Icon";
import MediaViewer from "./MediaViewer";
import PoseOverlay from "./PoseOverlay";

const WebcamWorkspace = forwardRef(function WebcamWorkspace({
  selectedBackend,
  modelPath,
  defaultModelPath,
  overlayOptions,
  onLiveChange,
  onPresentationChange,
}, ref) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const streamRef = useRef(null);
  const schedulerRef = useRef(null);
  const cameraGenerationRef = useRef(0);
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
    onLiveChange(false);
  }, [onLiveChange]);

  const stopCamera = useCallback(() => {
    cameraGenerationRef.current += 1;
    stopLive();
    stopMediaStream(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraStatus("idle");
    setCameraLabel("Camera");
    setCameraError(null);
    setLiveError(null);
    setResult(null);
  }, [stopLive]);

  useEffect(() => stopCamera, [stopCamera]);

  async function startCamera() {
    if (cameraStatus === "requesting" || cameraStatus === "active") return;
    const generation = cameraGenerationRef.current + 1;
    cameraGenerationRef.current = generation;
    setCameraStatus("requesting");
    setCameraError(null);
    try {
      const stream = await requestCamera(navigator.mediaDevices);
      if (generation !== cameraGenerationRef.current) {
        stopMediaStream(stream);
        return;
      }
      streamRef.current = stream;
      const [track] = stream.getVideoTracks();
      setCameraLabel(track?.label || "Camera");
      for (const currentTrack of stream.getVideoTracks()) currentTrack.addEventListener("ended", stopCamera, { once: true });
      if (generation === cameraGenerationRef.current) setCameraStatus("active");
    } catch (error) {
      if (generation !== cameraGenerationRef.current) return;
      stopMediaStream(streamRef.current);
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraStatus("error");
      setCameraError(normalizeCameraError(error));
    }
  }

  useEffect(() => {
    if (cameraStatus !== "active") return undefined;

    const video = videoRef.current;
    const stream = streamRef.current;
    if (!video || !stream) return undefined;

    let cancelled = false;
    video.srcObject = stream;
    video.play().catch((error) => {
      if (cancelled) return;
      stopMediaStream(streamRef.current);
      streamRef.current = null;
      video.srcObject = null;
      setCameraStatus("error");
      setCameraError(normalizeCameraError(error));
    });

    return () => {
      cancelled = true;
    };
  }, [cameraStatus]);

  async function captureFrame() {
    const captured = await captureMediaFrame(videoRef.current, canvasRef.current, contextRef.current, {
      notReadyCode: "camera_not_ready",
      notReadyMessage: "The camera is not ready to capture a frame.",
    });
    contextRef.current = captured.context;
    return captured.frameData;
  }

  function startLive() {
    if (cameraStatus !== "active" || !selectedBackend || schedulerRef.current) return;
    setLiveError(null);
    setResult(null);
    setLiveStatus("running");
    onLiveChange(true);
    const scheduler = createLiveScheduler({
      captureFrame,
      estimateFrame: (frameData) => estimatePoseFrame({ backend: selectedBackend, frameData, modelPath }),
      onResult: (nextResult) => {
        if (!validatePoseResult(nextResult)) throw { code: "unexpected_result", message: "The pose engine returned an unexpected live result." };
        setResult(nextResult);
      },
      onError: (error) => {
        schedulerRef.current = null;
        setLiveStatus("error");
        setLiveError(withModelSetupGuidance(error, selectedBackend, defaultModelPath));
        onLiveChange(false);
      },
      onTiming: (durationMs, completedAt) => {
        const previous = lastCompletionRef.current;
        const elapsed = previous === null ? durationMs : completedAt - previous;
        lastCompletionRef.current = completedAt;
        setInferenceFps(elapsed > 0 ? 1000 / elapsed : null);
      },
    });
    schedulerRef.current = scheduler;
    scheduler.start();
  }

  useImperativeHandle(ref, () => ({ startCamera, stopCamera, startLive, stopLive }));

  useEffect(() => {
    onPresentationChange?.({ cameraStatus, cameraLabel, liveStatus, result, inferenceFps, cameraError, liveError });
  }, [cameraStatus, cameraLabel, liveStatus, result, inferenceFps, cameraError, liveError, onPresentationChange]);

  const cameraActive = cameraStatus === "active";
  const liveActive = liveStatus === "running";
  const peopleCount = result?.people?.length ?? 0;
  const keypointCount = result?.people?.reduce((total, person) => total + (person.keypoints?.length ?? 0), 0) ?? 0;

  return (
    <div className="mode-workspace">
      <header className="workspace-heading">
        <div className="workspace-heading-contents">
          <span className="workspace-heading-title">Live Camera Analysis</span>
          <span className="workspace-heading-subtitle">{cameraActive ? cameraLabel : "Camera is off"}</span>
        </div>

        {liveActive &&
          <span className="status-chip success">
            <i />Live estimation active
          </span>
        }
      </header>

      {!cameraActive ? (
        <div className="media-stage empty-stage">
          <div className="empty-visual">
            <Icon name="camera" size={36} />
          </div>

          <div className="empty-stage-contents">
            <span className="empty-stage-title">Start the camera</span>
            <span className="empty-stage-subtitle">Camera access begins only when you choose Start Camera.</span>
          </div>

          <button
            className="primary-button compact-button"
            type="button"
            onClick={startCamera}
            disabled={cameraStatus === "requesting"}
          >
            <Icon name="camera" />
            {cameraStatus === "requesting" ? "Requesting Camera…" : "Start Camera"}
          </button>

          {cameraStatus === "requesting" && <div className="processing-overlay" role="status"><span className="spinner" /><strong>Requesting camera access…</strong></div>}
        </div>
      ) : (
        <MediaViewer className="webcam-viewer">
          <div className="media-stage live-media-stage">
          <figure className={`media-content-stage webcam-stage ${cameraActive ? "active" : ""}`}>
            <video ref={videoRef} muted playsInline aria-label="Live camera preview" />
            {cameraActive && result && <PoseOverlay result={result} showSkeleton={overlayOptions.skeleton} showKeypoints={overlayOptions.keypoints} showBoxes={overlayOptions.boxes} skeletonStyle={overlayOptions.skeletonStyle} lineThickness={overlayOptions.lineThickness} keypointSize={overlayOptions.keypointSize} />}
          </figure>
          {cameraActive && <button className="stage-corner-action" type="button" onClick={stopCamera}><Icon name="cameraOff" size={14} />Stop Camera</button>}
          {cameraActive && <div className="stage-badges"><span className="stage-badge">{result ? `${backendLabel(result.backend)} · ${formatPeopleCount(peopleCount)} detected · ${keypointCount} keypoints` : "Camera ready for pose analysis"}</span><span className="stage-badge success"><i />Camera active</span>{Number.isFinite(inferenceFps) && <span className="stage-badge success fps"><i />Inference {inferenceFps.toFixed(1)} FPS</span>}</div>}
          </div>
        </MediaViewer>
      )}
      {cameraError && <div className="workspace-error" role="alert"><strong>Camera input error</strong><p>{cameraError.message}</p></div>}
      {liveError && <div className="workspace-error" role="alert"><strong>Pose analysis error</strong><p>{liveError.message}</p></div>}
      {liveActive && result && peopleCount === 0 && <div className="zero-result">No people detected. Live inference is still running.</div>}
      <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
    </div>
  );
});

export default WebcamWorkspace;
