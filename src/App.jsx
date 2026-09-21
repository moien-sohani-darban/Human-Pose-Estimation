import { useCallback, useEffect, useState } from "react";
import {
  createPreviewSource,
  estimatePose,
  getBackends,
  selectImageFile,
  selectModelFile,
} from "./api/pose";
import ImageWorkspace from "./components/ImageWorkspace";
import PoseControls from "./components/PoseControls";
import ResultSummary from "./components/ResultSummary";
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

  async function chooseModel() {
    if (!selectedBackend || estimateStatus === "processing") return;

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
      !backendState.available.includes(backend)
    ) {
      return;
    }

    setSelectedBackend(backend);
    setPoseResult(null);
    setEstimateError(null);
    setEstimateStatus(selectedImage ? "ready" : "idle");
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
          <p>Local image pose analysis with MediaPipe and YOLO Pose</p>
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
          <ImageWorkspace
            selectedImage={selectedImage}
            result={poseResult}
            estimateStatus={estimateStatus}
            estimateError={estimateError}
            overlayOptions={overlayOptions}
            onChooseImage={chooseImage}
          />

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

          <ResultSummary result={poseResult} />
        </div>
      </div>
    </main>
  );
}

export default App;
