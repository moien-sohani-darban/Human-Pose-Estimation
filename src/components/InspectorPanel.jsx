import {
  backendLabel,
  getDefaultModelDisplayPath,
  getModelStatus,
} from "../utils/pose";
import CustomSelect from "./CustomSelect";
import Icon from "./Icon";
import ResultSummary from "./ResultSummary";

function Toggle({ checked, label, onChange }) {
  return (
    <label className="field-row toggle-row">
      <span className="field-row-label">{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="switch" aria-hidden="true" />
    </label>
  );
}

export default function InspectorPanel({
  backendState,
  selectedBackend,
  modelPath,
  controlsLocked,
  mode,
  selectedImage,
  selectedVideo,
  estimateStatus,
  result,
  inferenceFps,
  runtime,
  overlayOptions,
  compactOpen,
  onBackendChange,
  onChooseModel,
  onModelPathChange,
  onRetryBackends,
  onEstimate,
  onRuntimeAction,
}) {
  const metadata = backendState.models?.[selectedBackend];
  const modelStatus = getModelStatus(metadata, modelPath);
  const defaultPath = getDefaultModelDisplayPath(metadata);
  const defaultFile = defaultPath.split("/").at(-1) || "compatible model file";
  const isBusy = estimateStatus === "processing" || controlsLocked;
  const action = mode === "image"
    ? { label: estimateStatus === "processing" ? "Running Inference…" : "Run Inference", danger: false, disabled: !selectedImage || !selectedBackend || isBusy }
    : mode === "webcam"
      ? runtime?.liveStatus === "running"
        ? { label: "Stop Live Estimation", danger: true, disabled: false }
        : { label: "Start Live Estimation", danger: false, disabled: runtime?.cameraStatus !== "active" || !selectedBackend }
      : runtime?.analysisEnabled
        ? { label: "Stop Pose Analysis", danger: true, disabled: false }
        : { label: "Start Pose Analysis", danger: false, disabled: !selectedVideo || !selectedBackend || runtime?.mediaStatus === "loading" || runtime?.mediaStatus === "error" };

  return (
    <div className={`inspector-panel ${compactOpen ? "compact-open" : ""}`}>
      <div className="inspector-cards">
        <div className="inspector-card model-card" tabIndex="-1">
          <div className="card-heading">
            <div className="card-heading-left">
              <Icon name="model" />
              <span className="card-heading-title">Model &amp; Inference</span>
            </div>

            {modelStatus.kind === "ready" && (
              <span className="model-indicator ready">
                <i />
                Model ready
              </span>
            )}
          </div>

          <div className="field-row">
            <span className="field-row-label">Backend</span>

            <CustomSelect
              className="field-row-selectbox"
              ariaLabel="Backend"
              value={selectedBackend}
              onChange={onBackendChange}
              disabled={isBusy || backendState.status !== "ready"}
              options={[
                { value: "", label: "Choose backend", disabled: Boolean(selectedBackend) },
                ...backendState.available.map((backend) => ({ value: backend, label: backendLabel(backend) })),
                ...backendState.unavailable.map((backend) => ({ value: backend, label: `${backendLabel(backend)} (Unavailable)`, disabled: true })),
              ]}
            />
          </div>

          {backendState.status === "error" && (
            <div className="inline-panel-error">
              <span className="inline-panel-error-message">{backendState.error?.message}</span>
              <button
                className="inline-panel-error-button"
                type="button"
                onClick={onRetryBackends}
              >
                Retry
              </button>
            </div>
          )}

          <div className="field-row model-field">
            <span className="field-row-label">Model</span>

            <div className="model-input-group">
              <input
                className="model-input"
                value={modelPath}
                onChange={(event) => onModelPathChange(event.target.value)}
                placeholder={defaultFile}
                disabled={!selectedBackend || isBusy}
                spellCheck="false"
              />

              <button
                type="button"
                className="folder-button"
                onClick={onChooseModel}
                disabled={!selectedBackend || isBusy}
              >
                <Icon name="folder" size={12} />
              </button>
            </div>
          </div>

          {selectedBackend && modelStatus.kind === "missing" && (
            <div className="model-warning">
              <div className="warning-title-content">
                <Icon name="warning" size={16} />
                <span className="warning-title">Model required</span>
              </div>

              <div className="warning-description-content">
                {backendLabel(selectedBackend) ? (
                  <span className="warning-description">{backendLabel(selectedBackend)} is available, but its default model file is missing.</span>
                ) : (
                  <span className="warning-description">Model is unavailable, please choose a model.</span>
                )}

                <span className="warning-subtitle">Expected default: {defaultFile}</span>
              </div>

              <button
                className="primary-button"
                type="button"
                onClick={onChooseModel}
              >
                Choose Model
              </button>
            </div>
          )}

          {selectedBackend && modelStatus.kind !== "missing" && defaultPath && (
            <span className="model-hint">Default: {defaultPath}</span>
          )}

          {modelStatus.kind !== "missing" && (
            <div className="inference-actions">
              <button
                className={action.danger ? "danger-button" : "primary-button"}
                type="button"
                onClick={mode === "image" ? onEstimate : onRuntimeAction}
                disabled={action.disabled}
              >
                <Icon name={action.danger ? "stop" : "play"} size={12} />
                {action.label}
              </button>

              <button
                className="icon-button"
                type="button"
                onClick={onRetryBackends}
                disabled={isBusy}
              >
                <Icon name="refresh" />
              </button>
            </div>
          )}
        </div>

        <div className="inspector-card visualization-card">
          <div className="card-heading">
            <div className="card-heading-left">
              <Icon name="visual" />
              <span className="card-heading-title">Visualization</span>
            </div>
          </div>

          <Toggle
            label="Show Skeleton"
            checked={overlayOptions.skeleton}
            onChange={(value) => onRuntimeAction("overlay", "skeleton", value)}
          />

          <Toggle
            label="Show Keypoints"
            checked={overlayOptions.keypoints}
            onChange={(value) => onRuntimeAction("overlay", "keypoints", value)}
          />

          <Toggle
            label="Show Bounding Boxes"
            checked={overlayOptions.boxes}
            onChange={(value) => onRuntimeAction("overlay", "boxes", value)}
          />

          <div className="field-row">
            <span className="field-row-label">Skeleton Style</span>

            <CustomSelect
              className="skeleton-select"
              value={overlayOptions.skeletonStyle}
              options={[
                { value: "multi", label: "Multi-color limbs" },
                { value: "single", label: "Single color" },
              ]}
              onChange={(value) => onRuntimeAction("overlay", "skeletonStyle", value)}
            />
          </div>

          <div className="field-row range-row">
            <span className="field-row-label">Line Thickness</span>
            <input
              className="range-row-slider"
              type="range"
              min="1"
              max="5"
              step="0.5"
              value={overlayOptions.lineThickness}
              onChange={(event) => onRuntimeAction("overlay", "lineThickness", Number(event.target.value))}
            />
            <span className="range-row-slider-value">{overlayOptions.lineThickness}</span>
          </div>

          <div className="field-row range-row">
            <span className="field-row-label">Keypoint Size</span>
            <input
              className="range-row-slider"
              type="range"
              min="3"
              max="12"
              step="1"
              value={overlayOptions.keypointSize}
              onChange={(event) => onRuntimeAction("overlay", "keypointSize", Number(event.target.value))}
            />
            <span className="range-row-slider-value">{overlayOptions.keypointSize}</span>
          </div>
        </div>

        <ResultSummary result={result} inferenceFps={inferenceFps} mode={mode} />
      </div>
    </div>
  );
}
