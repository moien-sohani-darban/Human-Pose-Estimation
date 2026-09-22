import { useCallback, useEffect, useRef, useState } from "react";
import packageMetadata from "../package.json";
import {
  createPreviewSource,
  estimatePose,
  getBackends,
  selectImageFile,
  selectVideoFile,
  selectModelFile,
} from "./api/pose";
import ImageWorkspace from "./components/ImageWorkspace";
import PoseControls from "./components/PoseControls";
import ResultSummary from "./components/ResultSummary";
import WebcamWorkspace from "./components/WebcamWorkspace";
import VideoWorkspace from "./components/VideoWorkspace";
import { createVideoSelection } from "./utils/videoPose";
import {
  fileNameFromPath,
  getDefaultModelDisplayPath,
  normalizeModelMetadata,
  setBackendModelPath,
  validatePoseResult,
  withModelSetupGuidance,
} from "./utils/pose";
import "./App.css";

const initialBackendState = {
  status: "loading",
  available: [],
  unavailable: [],
  models: {},
  error: null,
};

function App() {
  const [backendState, setBackendState] = useState(initialBackendState);
  const [selectedBackend, setSelectedBackend] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [inputMode, setInputMode] = useState("image");
  const [liveLocked, setLiveLocked] = useState(false);
  const [webcamPresentation, setWebcamPresentation] = useState({
    cameraStatus: "idle",
    liveStatus: "idle",
    result: null,
    inferenceFps: null,
  });
  const [videoPresentation, setVideoPresentation] = useState({
    analysisEnabled: false,
    mediaStatus: "idle",
    result: null,
    inferenceFps: null,
    metadata: {
      duration: 0,
      width: 0,
      height: 0,
      currentTime: 0,
    },
    mediaError: null,
    analysisError: null,
  });
  const webcamRef = useRef(null);
  const videoRef = useRef(null);
  const [modelPaths, setModelPaths] = useState({
    mediapipe: "",
    yolo: "",
  });
  const [estimateStatus, setEstimateStatus] = useState("idle");
  const [poseResult, setPoseResult] = useState(null);
  const [estimateError, setEstimateError] = useState(null);
  const [interactionError, setInteractionError] = useState(null);
  const [overlayOptions, setOverlayOptions] = useState({
    skeleton: true,
    keypoints: true,
    boxes: true,
  });

  const loadBackends = useCallback(async () => {
    setBackendState((current) => ({
      ...current,
      status: "loading",
      error: null,
    }));

    try {
      const response = await getBackends();

      if (
        !Array.isArray(response?.available) ||
        !Array.isArray(response?.unavailable)
      ) {
        throw {
          code: "unexpected_result",
          message: "The pose engine returned an unexpected backend list.",
        };
      }

      setBackendState({
        status: "ready",
        available: response.available,
        unavailable: response.unavailable,
        models: normalizeModelMetadata(response.models),
        error: null,
      });

      setSelectedBackend((current) =>
        response.available.includes(current)
          ? current
          : response.available.includes("mediapipe")
            ? "mediapipe"
            : response.available[0] ?? "",
      );
    } catch (error) {
      setSelectedBackend("");
      setBackendState({
        status: "error",
        available: [],
        unavailable: [],
        models: {},
        error,
      });
    }
  }, []);

  useEffect(() => {
    loadBackends();
  }, [loadBackends]);

  async function chooseImage() {
    if (estimateStatus === "processing") return;

    setInteractionError(null);

    try {
      const path = await selectImageFile();
      if (!path) return;

      setSelectedImage({
        path,
        name: fileNameFromPath(path),
        previewSource: createPreviewSource(path),
      });

      setPoseResult(null);
      setEstimateError(null);
      setEstimateStatus("ready");
    } catch (error) {
      setInteractionError(error);
    }
  }

  async function chooseVideo() {
    if (liveLocked) return;

    setInteractionError(null);

    try {
      const path = await selectVideoFile();
      if (!path) return;

      setSelectedVideo((current) =>
        createVideoSelection(
          current,
          path,
          createPreviewSource,
        ),
      );
    } catch (error) {
      setInteractionError(error);
    }
  }

  async function chooseModel() {
    if (
      !selectedBackend ||
      estimateStatus === "processing" ||
      liveLocked
    ) {
      return;
    }

    setInteractionError(null);

    try {
      const path = await selectModelFile(selectedBackend);
      if (!path) return;

      setModelPaths((current) =>
        setBackendModelPath(current, selectedBackend, path),
      );
    } catch (error) {
      setInteractionError(error);
    }
  }

  function changeBackend(backend) {
    if (
      estimateStatus === "processing" ||
      liveLocked ||
      !backendState.available.includes(backend)
    ) {
      return;
    }

    setSelectedBackend(backend);
    setPoseResult(null);
    setEstimateError(null);
    setEstimateStatus(selectedImage ? "ready" : "idle");
  }

  function changeInputMode(nextMode) {
    if (nextMode === inputMode) return;

    if (inputMode === "webcam") {
      webcamRef.current?.stopCamera();
    }

    if (inputMode === "video") {
      videoRef.current?.stopAnalysis();
    }

    setLiveLocked(false);
    setInputMode(nextMode);
  }

  function clearImage() {
    if (estimateStatus === "processing") return;

    setSelectedImage(null);
    setPoseResult(null);
    setEstimateError(null);
    setInteractionError(null);
    setEstimateStatus("idle");
  }

  const selectedDefaultModelPath = getDefaultModelDisplayPath(
    backendState.models?.[selectedBackend],
  );

  async function runEstimate() {
    if (
      !selectedImage ||
      !selectedBackend ||
      estimateStatus === "processing"
    ) {
      return;
    }

    setPoseResult(null);
    setEstimateError(null);
    setInteractionError(null);
    setEstimateStatus("processing");

    try {
      const result = await estimatePose({
        backend: selectedBackend,
        imagePath: selectedImage.path,
        modelPath: modelPaths[selectedBackend] ?? "",
      });

      if (!validatePoseResult(result)) {
        throw {
          code: "unexpected_result",
          message: "The pose engine returned an unexpected result.",
        };
      }

      setPoseResult(result);
      setEstimateStatus("success");
    } catch (error) {
      setPoseResult(null);
      setEstimateError(
        withModelSetupGuidance(
          error,
          selectedBackend,
          selectedDefaultModelPath,
        ),
      );
      setEstimateStatus("error");
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1>Human Pose Estimation</h1>
          <p>Local image, webcam, and video pose analysis with MediaPipe and YOLO Pose</p>
        </div>

        <span
          className={`engine-status engine-${backendState.status}`}
          role="status"
        >
          {backendState.status === "loading"
            ? "Connecting…"
            : backendState.status === "ready"
              ? "Engine ready"
              : "Engine unavailable"}
        </span>
      </header>

      {interactionError && (
        <div className="global-error" role="alert">
          <span>{interactionError.message}</span>
          <button
            type="button"
            onClick={() => setInteractionError(null)}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}

      <div className="app-layout">
        <PoseControls
          backendState={backendState}
          selectedBackend={selectedBackend}
          selectedImage={selectedImage}
          modelPath={modelPaths[selectedBackend] ?? ""}
          estimateStatus={estimateStatus}
          liveLocked={liveLocked}
          inputMode={inputMode}
          onBackendChange={changeBackend}
          onChooseImage={chooseImage}
          onChooseModel={chooseModel}
          onEstimate={runEstimate}
          onRetryBackends={loadBackends}
          onModelPathChange={(value) =>
            setModelPaths((current) =>
              setBackendModelPath(current, selectedBackend, value),
            )
          }
          onClearImage={clearImage}
        />

        <div className="content-column">
          <div
            className="input-mode-switch"
            role="tablist"
            aria-label="Input mode"
          >
            <button
              type="button"
              className={inputMode === "image" ? "active" : ""}
              onClick={() => changeInputMode("image")}
              disabled={liveLocked}
            >
              Image
            </button>

            <button
              type="button"
              className={inputMode === "webcam" ? "active" : ""}
              onClick={() => changeInputMode("webcam")}
            >
              Webcam
            </button>

            <button
              type="button"
              className={inputMode === "video" ? "active" : ""}
              onClick={() => changeInputMode("video")}
              disabled={liveLocked}
            >
              Video
            </button>
          </div>

          {inputMode === "image" ? (
            <ImageWorkspace
              selectedImage={selectedImage}
              result={poseResult}
              estimateStatus={estimateStatus}
              estimateError={estimateError}
              overlayOptions={overlayOptions}
              onChooseImage={chooseImage}
            />
          ) : inputMode === "webcam" ? (
            <WebcamWorkspace
              ref={webcamRef}
              selectedBackend={selectedBackend}
              modelPath={modelPaths[selectedBackend] ?? ""}
              defaultModelPath={selectedDefaultModelPath}
              overlayOptions={overlayOptions}
              onLiveChange={setLiveLocked}
              onPresentationChange={setWebcamPresentation}
            />
          ) : (
            <VideoWorkspace
              ref={videoRef}
              selectedVideo={selectedVideo}
              selectedBackend={selectedBackend}
              modelPath={modelPaths[selectedBackend] ?? ""}
              defaultModelPath={selectedDefaultModelPath}
              overlayOptions={overlayOptions}
              onAnalysisChange={setLiveLocked}
              onPresentationChange={setVideoPresentation}
              onChooseVideo={chooseVideo}
            />
          )}

          <div className="overlay-controls">
            <label>
              <input
                type="checkbox"
                checked={overlayOptions.skeleton}
                onChange={(event) =>
                  setOverlayOptions((current) => ({
                    ...current,
                    skeleton: event.target.checked,
                  }))
                }
              />
              Skeleton
            </label>

            <label>
              <input
                type="checkbox"
                checked={overlayOptions.keypoints}
                onChange={(event) =>
                  setOverlayOptions((current) => ({
                    ...current,
                    keypoints: event.target.checked,
                  }))
                }
              />
              Keypoints
            </label>

            <label>
              <input
                type="checkbox"
                checked={overlayOptions.boxes}
                onChange={(event) =>
                  setOverlayOptions((current) => ({
                    ...current,
                    boxes: event.target.checked,
                  }))
                }
              />
              Bounding boxes
            </label>
          </div>

          <ResultSummary
            result={
              inputMode === "webcam"
                ? webcamPresentation.result
                : inputMode === "video"
                  ? videoPresentation.result
                  : poseResult
            }
          />
        </div>
      </div>

      <footer className="app-footer">
        <span>Human Pose Estimation v{packageMetadata.version}</span>
        <span>Local processing | MediaPipe + YOLO Pose</span>
      </footer>
    </main>
  );
}

export default App;
