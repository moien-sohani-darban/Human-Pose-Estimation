import { backendLabel } from "../utils/pose";

const BACKEND_DESCRIPTIONS = {
  mediapipe: "Detailed canonical 33-landmark pose estimation.",
  yolo: "Fast multi-person pose detection.",
};

export default function PoseControls({
  backendState,
  selectedBackend,
  selectedImage,
  modelPath,
  estimateStatus,
  onBackendChange,
  onChooseImage,
  onChooseModel,
  onEstimate,
  onRetryBackends,
  onModelPathChange,
  onClearImage,
}) {
  const isProcessing = estimateStatus === "processing";

  const canEstimate = Boolean(
    selectedImage &&
      selectedBackend &&
      backendState.available.includes(selectedBackend) &&
      !isProcessing,
  );

  return (
    <aside className="control-panel" aria-label="Pose estimation controls">
      <section className="control-section">
        <div className="section-heading">
          <span className="step-number">01</span>
          <div>
            <h2>Input image</h2>
            <p>PNG, JPG, JPEG, BMP, or WebP</p>
          </div>
        </div>

        <button
          type="button"
          className="secondary-button full-width"
          onClick={onChooseImage}
          disabled={isProcessing}
        >
          {selectedImage ? "Choose another image" : "Choose image"}
        </button>

        {selectedImage && (
          <div className="selected-file">
            <div>
              <span>Selected</span>
              <strong title={selectedImage.path}>{selectedImage.name}</strong>
            </div>

            <button
              type="button"
              className="text-button"
              onClick={onClearImage}
              disabled={isProcessing}
            >
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
            <p>Availability comes from the local Python engine</p>
          </div>

          <button
            type="button"
            className="icon-button"
            onClick={onRetryBackends}
            disabled={backendState.status === "loading" || isProcessing}
            aria-label="Refresh backend availability"
            title="Refresh backend availability"
          >
            ↻
          </button>
        </div>

        {backendState.status === "loading" && (
          <div className="inline-status" role="status">
            Starting pose engine…
          </div>
        )}

        {backendState.status === "error" && (
          <div className="inline-error" role="alert">
            <p>{backendState.error?.message}</p>
            <button
              type="button"
              className="text-button"
              onClick={onRetryBackends}
            >
              Retry
            </button>
          </div>
        )}

        {backendState.status === "ready" && (
          <>
            {backendState.available.length > 0 ? (
              <div
                className="backend-list"
                role="radiogroup"
                aria-label="Backend"
              >
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
                      disabled={isProcessing}
                    />
                    <span>
                      <strong>{backendLabel(backend)}</strong>
                      <small>
                        {BACKEND_DESCRIPTIONS[backend] ??
                          "Available pose-estimation backend."}
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
              <p className="empty-backends">
                No pose backend is currently available.
              </p>
            )}

            {backendState.unavailable.map((backend) => (
              <div className="unavailable-backend" key={backend}>
                <span>{backendLabel(backend)}</span>
                <small>
                  Unavailable
                  {backend === "mmpose"
                    ? " in the current Python environment"
                    : ""}
                </small>
              </div>
            ))}
          </>
        )}
      </section>

      <section className="control-section">
        <div className="section-heading">
          <span className="step-number">03</span>
          <div>
            <h2>Model file</h2>
            <p>Optional development-time override</p>
          </div>
        </div>

        <label className="field-label" htmlFor="model-path">
          {selectedBackend
            ? `${backendLabel(selectedBackend)} model path`
            : "Model path"}
        </label>

        <div className="path-input-row">
          <input
            id="model-path"
            className="path-input"
            value={modelPath}
            onChange={(event) => onModelPathChange(event.target.value)}
            placeholder="Optional model path"
            disabled={!selectedBackend || isProcessing}
            spellCheck="false"
          />

          <button
            type="button"
            className="browse-button"
            onClick={onChooseModel}
            disabled={!selectedBackend || isProcessing}
          >
            Browse
          </button>
        </div>

        <p className="field-hint">
          Leave blank to use the backend default.
        </p>
      </section>

      <button
        type="button"
        className="primary-button"
        onClick={onEstimate}
        disabled={!canEstimate}
      >
        {isProcessing ? "Running pose estimation…" : "Estimate Pose"}
      </button>
    </aside>
  );
}
