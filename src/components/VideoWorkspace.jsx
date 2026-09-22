import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { estimatePoseFrame } from "../api/pose";
import { captureMediaFrame } from "../utils/livePose";
import {
  createVideoAnalysisController,
  formatVideoTime,
} from "../utils/videoPose";
import {
  backendLabel,
  formatPeopleCount,
  validatePoseResult,
  withModelSetupGuidance,
} from "../utils/pose";
import PoseOverlay from "./PoseOverlay";

const VideoWorkspace = forwardRef(function VideoWorkspace(
  {
    selectedVideo,
    selectedBackend,
    modelPath,
    defaultModelPath,
    overlayOptions,
    onAnalysisChange,
    onPresentationChange,
    onChooseVideo,
  },
  ref,
) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const controllerRef = useRef(null);
  const lastCompletionRef = useRef(null);

  const [mediaStatus, setMediaStatus] = useState(
    selectedVideo ? "loading" : "idle",
  );
  const [mediaError, setMediaError] = useState(null);
  const [analysisEnabled, setAnalysisEnabled] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const [result, setResult] = useState(null);
  const [inferenceFps, setInferenceFps] = useState(null);
  const [metadata, setMetadata] = useState({
    duration: 0,
    width: 0,
    height: 0,
    currentTime: 0,
  });

  const clearPoseState = useCallback(() => {
    setResult(null);
    setInferenceFps(null);
    lastCompletionRef.current = null;
  }, []);

  const stopAnalysis = useCallback(() => {
    controllerRef.current?.stop();
    controllerRef.current = null;
    setAnalysisEnabled(false);
    setAnalysisError(null);
    clearPoseState();
    onAnalysisChange(false);
  }, [clearPoseState, onAnalysisChange]);

  useEffect(() => {
    controllerRef.current?.replaceSource();
    controllerRef.current = null;

    setAnalysisEnabled(false);
    setAnalysisError(null);
    setMediaError(null);
    setMediaStatus(selectedVideo ? "loading" : "idle");
    setMetadata({
      duration: 0,
      width: 0,
      height: 0,
      currentTime: 0,
    });
    clearPoseState();
    onAnalysisChange(false);

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.load();
    }

    return () => {
      controllerRef.current?.replaceSource();
      controllerRef.current = null;
      videoRef.current?.pause();
      onAnalysisChange(false);
    };
  }, [selectedVideo?.path, clearPoseState, onAnalysisChange]);

  async function captureCurrentFrame() {
    const captured = await captureMediaFrame(
      videoRef.current,
      canvasRef.current,
      contextRef.current,
      {
        notReadyCode: "video_not_ready",
        notReadyMessage:
          "The video is not ready to capture a pose frame.",
      },
    );

    contextRef.current = captured.context;
    return captured.frameData;
  }

  function startAnalysis() {
    const video = videoRef.current;

    if (
      !video ||
      !selectedBackend ||
      controllerRef.current ||
      mediaStatus === "error"
    ) {
      return;
    }

    setAnalysisError(null);
    clearPoseState();
    setAnalysisEnabled(true);
    onAnalysisChange(true);

    const controller = createVideoAnalysisController({
      media: video,
      captureCurrentFrame,
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
            message:
              "The pose engine returned an unexpected video result.",
          };
        }

        setResult(nextResult);
      },
      onError: (error) => {
        controllerRef.current = null;
        setAnalysisEnabled(false);
        setAnalysisError(
          withModelSetupGuidance(
            error,
            selectedBackend,
            defaultModelPath,
          ),
        );
        clearPoseState();
        onAnalysisChange(false);
      },
      onInvalidate: clearPoseState,
      onEnded: () => {
        controllerRef.current = null;
        setAnalysisEnabled(false);
        onAnalysisChange(false);
      },
      onTiming: (durationMs, completedAt) => {
        const previous = lastCompletionRef.current;
        const elapsed =
          previous === null ? durationMs : completedAt - previous;

        lastCompletionRef.current = completedAt;
        setInferenceFps(elapsed > 0 ? 1000 / elapsed : null);
      },
    });

    controllerRef.current = controller;
    controller.start();
  }

  useImperativeHandle(ref, () => ({
    startAnalysis,
    stopAnalysis,
  }));

  useEffect(() => {
    onPresentationChange?.({
      analysisEnabled,
      mediaStatus,
      result,
      inferenceFps,
      metadata,
      mediaError,
      analysisError,
    });
  }, [
    analysisEnabled,
    mediaStatus,
    result,
    inferenceFps,
    metadata,
    mediaError,
    analysisError,
    onPresentationChange,
  ]);

  function updateMetadata(video) {
    setMetadata((current) => ({
      ...current,
      duration: Number.isFinite(video.duration) ? video.duration : 0,
      width: video.videoWidth,
      height: video.videoHeight,
      currentTime: Number.isFinite(video.currentTime)
        ? video.currentTime
        : 0,
    }));
  }

  const peopleCount = result?.people?.length ?? 0;

  if (!selectedVideo) {
    return (
      <section className="video-workspace">
        <div className="video-empty-state">
          <h2>Video pose estimation</h2>
          <p>
            Choose a local MP4, WebM, MOV, or M4V file to begin.
          </p>

          <button
            type="button"
            className="primary-button"
            onClick={onChooseVideo}
          >
            Choose Video
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="video-workspace">
      <div className="video-stage">
        <div className="video-frame">
          <video
            ref={videoRef}
            src={selectedVideo.previewSource}
            controls
            playsInline
            preload="metadata"
            aria-label={`Selected video: ${selectedVideo.name}`}
            onLoadedMetadata={(event) => {
              updateMetadata(event.currentTarget);
              setMediaStatus("ready");
            }}
            onCanPlay={() =>
              setMediaStatus((current) =>
                current === "loading" ? "ready" : current,
              )
            }
            onPlay={() => {
              setMediaStatus("playing");
              controllerRef.current?.handlePlay();
            }}
            onPause={(event) => {
              if (
                !event.currentTarget.ended &&
                !event.currentTarget.seeking
              ) {
                setMediaStatus("paused");
              }

              controllerRef.current?.handlePause();
            }}
            onSeeking={() => {
              setMediaStatus("seeking");
              controllerRef.current?.handleSeeking();
            }}
            onSeeked={(event) => {
              updateMetadata(event.currentTarget);
              setMediaStatus(
                event.currentTarget.paused ? "paused" : "playing",
              );
              controllerRef.current?.handleSeeked();
            }}
            onTimeUpdate={(event) =>
              updateMetadata(event.currentTarget)
            }
            onEnded={() => {
              setMediaStatus("ended");
              controllerRef.current?.handleEnded();
            }}
            onError={() => {
              setMediaStatus("error");
              setMediaError({
                code: "video_playback_failed",
                message:
                  "This video could not be played. Choose a supported local video file.",
              });

              controllerRef.current?.stop();
              controllerRef.current = null;
              setAnalysisEnabled(false);
              onAnalysisChange(false);
              clearPoseState();
            }}
          />

          {result && (
            <PoseOverlay
              result={result}
              showSkeleton={overlayOptions.skeleton}
              showKeypoints={overlayOptions.keypoints}
              showBoxes={overlayOptions.boxes}
            />
          )}
        </div>

        <div className="video-meta-row">
          <span>
            {result
              ? `${backendLabel(result.backend)} ? ${formatPeopleCount(
                  peopleCount,
                )} detected`
              : `Video ${mediaStatus}`}
          </span>

          {metadata.duration > 0 && (
            <span>
              {formatVideoTime(metadata.currentTime)} /{" "}
              {formatVideoTime(metadata.duration)}
            </span>
          )}

          {Number.isFinite(inferenceFps) && (
            <span>Inference {inferenceFps.toFixed(1)} FPS</span>
          )}
        </div>
      </div>

      <div className="video-actions">
        <button
          type="button"
          className="secondary-button"
          onClick={onChooseVideo}
          disabled={analysisEnabled}
        >
          Choose another video
        </button>

        {analysisEnabled ? (
          <button
            type="button"
            className="secondary-button"
            onClick={stopAnalysis}
          >
            Stop Analysis
          </button>
        ) : (
          <button
            type="button"
            className="primary-button"
            onClick={startAnalysis}
            disabled={
              mediaStatus === "loading" ||
              mediaStatus === "error" ||
              !selectedBackend
            }
          >
            Start Analysis
          </button>
        )}
      </div>

      {mediaError && (
        <div className="workspace-error" role="alert">
          <strong>Video playback error</strong>
          <p>{mediaError.message}</p>
        </div>
      )}

      {analysisError && (
        <div className="workspace-error" role="alert">
          <strong>Pose analysis error</strong>
          <p>{analysisError.message}</p>
        </div>
      )}

      {analysisEnabled && result && peopleCount === 0 && (
        <div className="zero-result" role="status">
          No people detected. Video analysis is still enabled.
        </div>
      )}

      <canvas
        ref={canvasRef}
        className="capture-canvas"
        aria-hidden="true"
      />
    </section>
  );
});

export default VideoWorkspace;
