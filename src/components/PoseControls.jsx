import {
  backendLabel,
  getDefaultModelDisplayPath,
  getModelStatus,
} from "../utils/pose";

const BACKEND_DESCRIPTIONS = {
  mediapipe: "Detailed canonical 33-landmark estimation.",
  yolo: "Fast multi-person pose detection.",
};

export default function PoseControls({
  backendState,
  selectedBackend,
  selectedImage,
  selectedVideo,
  modelPath,
  estimateStatus,
  inputMode,
  controlsLocked,
  onBackendChange,
  onChooseImage,
  onChooseModel,
  onChooseVideo,
  onEstimate,
  onRetryBackends,
  onModelPathChange,
  onClearImage,
  onClearVideo,
}) {
  const isProcessing = estimateStatus === "processing";
  const isLocked = isProcessing || controlsLocked;
  const canEstimate = Boolean(
    inputMode === "image" && selectedImage && selectedBackend && !isLocked,
  );
  const selectedModelMetadata = backendState.models?.[selectedBackend];
  const selectedModelStatus = getModelStatus(selectedModelMetadata, modelPath);
  const defaultDisplayPath = getDefaultModelDisplayPath(selectedModelMetadata);

  return (
    <aside className="control-panel" aria-label="Pose estimation controls">
      <section className="control-section">
        <div className="section-heading">
          <span className="step-number">01</span>
          <div>
            <h2>{inputMode === "image" ? "Input image" : inputMode === "webcam" ? "Live camera" : "Input video"}</h2>
            <p>{inputMode === "image" ? "PNG, JPG, JPEG, BMP, or WebP" : inputMode === "webcam" ? "Camera controls are in the live workspace" : "MP4, WebM, MOV, or M4V"}</p>
          </div>
        </div>
        {inputMode === "image" && <button
          className="secondary-button full-width"
          onClick={onChooseImage}
          disabled={isLocked}
        >
          {selectedImage ? "Choose another image" : "Choose image"}
        </button>}
        {inputMode === "image" && selectedImage && (
          <div className="selected-file">
            <div>
              <span>Selected</span>
              <strong title={selectedImage.path}>{selectedImage.name}</strong>
            </div>
            <button className="text-button" onClick={onClearImage} disabled={isLocked}>
              Clear
            </button>
          </div>
        )}
        {inputMode === "video" && <button
          className="secondary-button full-width"
          onClick={onChooseVideo}
          disabled={isLocked}
        >
          {selectedVideo ? "Choose another video" : "Choose video"}
        </button>}
        {inputMode === "video" && selectedVideo && (
          <div className="selected-file">
            <div>
              <span>Selected</span>
              <strong title={selectedVideo.path}>{selectedVideo.name}</strong>
            </div>
            <button className="text-button" onClick={onClearVideo} disabled={isLocked}>
              Clear
            </button>
          </div>
        )}
      </section>

      <section className="control-section">
        <div className="section-heading backend-heading">
          <span className="step-number">02</span>
          <div>
            <h2>Pose backend</h2>
            <p>Availability comes from the Python engine</p>
          </div>
          <button
            className="icon-button"
            onClick={onRetryBackends}
            disabled={backendState.status === "loading" || isLocked}
            aria-label="Refresh backend availability"
            title="Refresh backend availability"
          >
            ↻
          </button>
        </div>

        {backendState.status === "loading" && (
          <div className="inline-status" role="status">
            <span className="spinner small" /> Starting pose engine…
          </div>
        )}

        {backendState.status === "error" && (
          <div className="inline-error" role="alert">
            <p>{backendState.error.message}</p>
            <button className="text-button" onClick={onRetryBackends}>
              Retry
            </button>
          </div>
        )}

        {backendState.status === "ready" && (
          <>
            {backendState.available.length > 0 ? (
              <div className="backend-list" role="radiogroup" aria-label="Backend">
                {backendState.available.map((backend) => (
                  <label
                    className={`backend-option ${
                      selectedBackend === backend ? "selected" : ""
                    }`}
                    key={backend}
                  >
                    <input
                      type="radio"
                      name="pose-backend"
                      value={backend}
                      checked={selectedBackend === backend}
                      onChange={() => onBackendChange(backend)}
                      disabled={isLocked}
                    />
                    <span>
                      <strong>{backendLabel(backend)}</strong>
                      <small>
                        {BACKEND_DESCRIPTIONS[backend] ??
                          "Available pose-estimation backend."}
                      </small>
                      <small className={`backend-model-state ${getModelStatus(
                        backendState.models?.[backend],
                        "",
                      ).kind}`}>
                        {getModelStatus(backendState.models?.[backend], "").label}
                      </small>
                    </span>
                    <span
                      className="availability-dot"
                      role="img"
                      aria-label="Available"
                    />
                  </label>
                ))}
              </div>
            ) : (
              <p className="empty-backends">No pose backend is currently available.</p>
            )}

            {backendState.unavailable.map((backend) => (
              <div className="unavailable-backend" key={backend}>
                <span>{backendLabel(backend)}</span>
                <small>
                  Unavailable
                  {backend === "mmpose" ? " in the current Python environment" : ""}
                </small>
              </div>
            ))}
          </>
        )}
      </section>

      <section className="control-section model-section">
        <div className="section-heading">
          <span className="step-number">03</span>
          <div>
            <h2>Model file</h2>
            <p>Use the default or select a compatible local model</p>
          </div>
        </div>
        <label className="field-label" htmlFor="model-path">
          {selectedBackend ? `${backendLabel(selectedBackend)} model path` : "Model path"}
        </label>
        <div className="path-input-row">
          <input
            id="model-path"
            className="path-input"
            value={modelPath}
            onChange={(event) => onModelPathChange(event.target.value)}
            placeholder={
              defaultDisplayPath
                ? `Default: ${defaultDisplayPath}`
                : "Optional custom model path"
            }
            disabled={!selectedBackend || isLocked}
            spellCheck="false"
          />
          <button
            className="browse-button"
            onClick={onChooseModel}
            disabled={!selectedBackend || isLocked}
          >
            Browse
          </button>
        </div>
        {selectedBackend && (
          <div
            className={`model-status model-status-${selectedModelStatus.kind}`}
            role="status"
          >
            <strong>{selectedModelStatus.label}</strong>
            <span>{selectedModelStatus.guidance}</span>
          </div>
        )}
        {defaultDisplayPath && (
          <p className="field-hint">
            Default: <code>{defaultDisplayPath}</code>
          </p>
        )}
        {!defaultDisplayPath && (
          <p className="field-hint">
            Leave blank to use the backend’s default local path.
          </p>
        )}
      </section>

      {inputMode === "image" && <button
        className="primary-button"
        onClick={onEstimate}
        disabled={!canEstimate}
      >
        {isProcessing ? (
          <>
            <span className="spinner" /> Running pose estimation…
          </>
        ) : (
          "Estimate Pose"
        )}
      </button>}
    </aside>
  );
}
