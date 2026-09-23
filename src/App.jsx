import { useCallback, useEffect, useRef, useState } from "react";
import {
  createPreviewSource,
  estimatePose,
  getBackends,
  selectImageFile,
  selectModelFile,
  selectVideoFile,
} from "./api/pose";
import Icon from "./components/Icon";
import ImageWorkspace from "./components/ImageWorkspace";
import InspectorPanel from "./components/InspectorPanel";
import NavigationSidebar from "./components/NavigationSidebar";
import SettingsModal from "./components/SettingsModal";
import WebcamWorkspace from "./components/WebcamWorkspace";
import VideoWorkspace from "./components/VideoWorkspace";
import {
  fileNameFromPath,
  getDefaultModelDisplayPath,
  normalizeModelMetadata,
  setBackendModelPath,
  validatePoseResult,
  withModelSetupGuidance,
} from "./utils/pose";
import { createVideoSelection } from "./utils/videoPose";
import {
  persistAppearance,
  readAppearancePreference,
  resolveTheme,
} from "./utils/theme";
import "./App.scss";

import HPEImage from "./assets/images/HPE-Logo.png";

const initialBackendState = { status: "loading", available: [], unavailable: [], models: {}, error: null };
const initialRuntime = {
  webcam: { cameraStatus: "idle", liveStatus: "idle", result: null, inferenceFps: null },
  video: { analysisEnabled: false, mediaStatus: "idle", result: null, inferenceFps: null },
};

function App() {
  const [backendState, setBackendState] = useState(initialBackendState);
  const [selectedBackend, setSelectedBackend] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [modelPaths, setModelPaths] = useState({ mediapipe: "", yolo: "" });
  const [estimateStatus, setEstimateStatus] = useState("idle");
  const [poseResult, setPoseResult] = useState(null);
  const [estimateError, setEstimateError] = useState(null);
  const [interactionError, setInteractionError] = useState(null);
  const [overlayOptions, setOverlayOptions] = useState({
    skeleton: true,
    keypoints: true,
    boxes: true,
    skeletonStyle: "single",
    lineThickness: 2,
    keypointSize: 6,
  });
  const [inputMode, setInputMode] = useState("image");
  const [liveLocked, setLiveLocked] = useState(false);
  const [runtime, setRuntime] = useState(initialRuntime);
  const [compactInspectorOpen, setCompactInspectorOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [appearance, setAppearance] = useState(() => readAppearancePreference(globalThis.localStorage));
  const [systemDark, setSystemDark] = useState(() => globalThis.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true);
  const webcamRef = useRef(null);
  const videoRef = useRef(null);
  const settingsTriggerRef = useRef(null);

  const selectedDefaultModelPath = getDefaultModelDisplayPath(backendState.models?.[selectedBackend]);

  const loadBackends = useCallback(async () => {
    setBackendState((current) => ({ ...current, status: "loading", error: null }));
    try {
      const response = await getBackends();
      if (!Array.isArray(response?.available) || !Array.isArray(response?.unavailable)) {
        throw { code: "unexpected_result", message: "The pose engine returned an unexpected backend list." };
      }
      setBackendState({ status: "ready", available: response.available, unavailable: response.unavailable, models: normalizeModelMetadata(response.models), error: null });
      setSelectedBackend((current) => response.available.includes(current) ? current : response.available.includes("mediapipe") ? "mediapipe" : response.available[0] ?? "");
    } catch (error) {
      setSelectedBackend("");
      setBackendState({ status: "error", available: [], unavailable: [], models: {}, error });
    }
  }, []);

  useEffect(() => { loadBackends(); }, [loadBackends]);

  useEffect(() => {
    const media = globalThis.matchMedia?.("(prefers-color-scheme: dark)");
    if (!media) return undefined;
    const update = (event) => setSystemDark(event.matches);
    media.addEventListener?.("change", update);
    return () => media.removeEventListener?.("change", update);
  }, []);

  useEffect(() => {
    globalThis.document?.documentElement?.setAttribute("data-theme", resolveTheme(appearance, systemDark));
    persistAppearance(globalThis.localStorage, appearance);
  }, [appearance, systemDark]);

  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
    globalThis.setTimeout?.(() => settingsTriggerRef.current?.focus(), 0);
  }, []);
  const handleWebcamPresentation = useCallback((next) => {
    setRuntime((current) => ({ ...current, webcam: next }));
  }, []);
  const handleVideoPresentation = useCallback((next) => {
    setRuntime((current) => ({ ...current, video: next }));
  }, []);

  async function chooseImage() {
    if (estimateStatus === "processing") return;
    setInteractionError(null);
    try {
      const path = await selectImageFile();
      if (!path) return;
      setSelectedImage({ path, name: fileNameFromPath(path), previewSource: createPreviewSource(path) });
      setPoseResult(null);
      setEstimateError(null);
      setEstimateStatus("ready");
    } catch (error) { setInteractionError(error); }
  }

  async function chooseModel() {
    if (!selectedBackend || liveLocked) return;
    setInteractionError(null);
    try {
      const path = await selectModelFile(selectedBackend);
      if (path) setModelPaths((current) => setBackendModelPath(current, selectedBackend, path));
    } catch (error) { setInteractionError(error); }
  }

  async function chooseVideo() {
    if (liveLocked) return;
    setInteractionError(null);
    try {
      const path = await selectVideoFile();
      if (path) setSelectedVideo((current) => createVideoSelection(current, path, createPreviewSource));
    } catch (error) { setInteractionError(error); }
  }

  function changeBackend(backend) {
    if (estimateStatus === "processing" || liveLocked || !backendState.available.includes(backend)) return;
    setSelectedBackend(backend);
    setPoseResult(null);
    setEstimateError(null);
    setEstimateStatus(selectedImage ? "ready" : "idle");
  }

  async function runEstimate() {
    if (!selectedImage || !selectedBackend || estimateStatus === "processing") return;
    setPoseResult(null);
    setEstimateError(null);
    setInteractionError(null);
    setEstimateStatus("processing");
    try {
      const result = await estimatePose({ backend: selectedBackend, imagePath: selectedImage.path, modelPath: modelPaths[selectedBackend] ?? "" });
      if (!validatePoseResult(result)) throw { code: "unexpected_result", message: "The pose engine returned an unexpected result." };
      setPoseResult(result);
      setEstimateStatus("success");
    } catch (error) {
      setPoseResult(null);
      setEstimateError(withModelSetupGuidance(error, selectedBackend, selectedDefaultModelPath));
      setEstimateStatus("error");
    }
  }

  function changeMode(nextMode) {
    if (nextMode === inputMode) return;
    if (inputMode === "webcam") webcamRef.current?.stopCamera();
    if (inputMode === "video") videoRef.current?.stopAnalysis();
    setLiveLocked(false);
    setCompactInspectorOpen(false);
    setInputMode(nextMode);
  }

  function handleInspectorAction(kind, key, value) {
    if (kind === "overlay") {
      setOverlayOptions((current) => ({ ...current, [key]: value }));
      return;
    }
    if (inputMode === "webcam") {
      if (runtime.webcam.liveStatus === "running") webcamRef.current?.stopLive();
      else webcamRef.current?.startLive();
    } else if (inputMode === "video") {
      if (runtime.video.analysisEnabled) videoRef.current?.stopAnalysis();
      else videoRef.current?.startAnalysis();
    }
  }

  function focusModelSetup() {
    setCompactInspectorOpen(true);
    globalThis.setTimeout?.(() => globalThis.document?.getElementById("model-setup")?.focus(), 0);
  }

  const activeRuntime = runtime[inputMode] ?? null;
  const activeResult = inputMode === "image" ? poseResult : activeRuntime?.result ?? null;
  const activeFps = inputMode === "image" && Number.isFinite(poseResult?.processing_time_ms) && poseResult.processing_time_ms > 0
    ? 1000 / poseResult.processing_time_ms
    : activeRuntime?.inferenceFps ?? null;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-logo">
            <img className="brand-image" src={HPEImage} alt="Logo" />
          </div>

          <div className="brand-contents">
            <span className="brand-title">Human Pose Estimation</span>
            <p className="brand-subtitle">Local pose analysis for images, videos and webcam</p>
          </div>
        </div>

        <button
          className="header-inspector-toggle"
          type="button"
          aria-label="Toggle inspector"
          aria-controls="inspector-panel"
          aria-expanded={compactInspectorOpen}
          onClick={() => setCompactInspectorOpen((current) => !current)}
        >
          <Icon name="preferences" />
        </button>
      </header>

      <div className="desktop-shell" inert={settingsOpen ? true : undefined}>
        <NavigationSidebar
          inputMode={inputMode}
          engineStatus={backendState.status}
          onModeChange={changeMode}
          onModelSetup={focusModelSetup}
          onPreferences={() => setSettingsOpen(true)}
        />

        <div className="main-workspace">
          {interactionError &&
            <div className="global-error" role="alert">
              <div>
                <strong>Action needed</strong>
                <span>{interactionError.message}</span>
              </div>

              <button
                type="button"
                onClick={() => setInteractionError(null)}
              >
                ×
              </button>
            </div>
          }

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
              onPresentationChange={handleWebcamPresentation}
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
              onPresentationChange={handleVideoPresentation}
              onChooseVideo={chooseVideo}
            />
          )}
        </div>

        <InspectorPanel
          backendState={backendState}
          selectedBackend={selectedBackend}
          modelPath={modelPaths[selectedBackend] ?? ""}
          controlsLocked={liveLocked}
          mode={inputMode}
          selectedImage={selectedImage}
          selectedVideo={selectedVideo}
          estimateStatus={estimateStatus}
          result={activeResult}
          inferenceFps={activeFps}
          runtime={activeRuntime}
          overlayOptions={overlayOptions}
          compactOpen={compactInspectorOpen}
          onBackendChange={changeBackend}
          onChooseModel={chooseModel}
          onModelPathChange={(value) => !liveLocked && setModelPaths((current) =>
            setBackendModelPath(current, selectedBackend, value))
          }
          onRetryBackends={() => !liveLocked && loadBackends()}
          onEstimate={runEstimate}
          onRuntimeAction={handleInspectorAction}
        />
      </div>

      <SettingsModal open={settingsOpen} appearance={appearance} onAppearanceChange={setAppearance} onClose={closeSettings} />
    </div>
  );
}

export default App;
