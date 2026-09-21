import { useEffect, useState } from "react";
import { backendLabel, formatPeopleCount } from "../utils/pose";
import PoseOverlay from "./PoseOverlay";

export default function ImageWorkspace({
  selectedImage,
  result,
  estimateStatus,
  estimateError,
  overlayOptions,
  onChooseImage,
}) {
  const [previewStatus, setPreviewStatus] = useState("empty");
  const [dimensions, setDimensions] = useState(null);

  useEffect(() => {
    setPreviewStatus(selectedImage ? "loading" : "empty");
    setDimensions(null);
  }, [selectedImage]);

  const people = Array.isArray(result?.people) ? result.people : [];

  if (!selectedImage) {
    return (
      <section className="workspace">
        <div className="empty-stage">
          <div className="empty-icon" aria-hidden="true">◇</div>
          <h2>Select an image to begin</h2>
          <p>Choose a local image and run pose estimation.</p>

          <button
            type="button"
            className="primary-button compact-button"
            onClick={onChooseImage}
          >
            Choose Image
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="workspace">
      <header className="workspace-heading">
        <div>
          <h2>{selectedImage.name}</h2>
          <p>
            {dimensions
              ? `${dimensions.width} × ${dimensions.height}`
              : "Image analysis"}
          </p>
        </div>

        {estimateStatus === "success" && (
          <span className="status-chip success">Analysis complete</span>
        )}
      </header>

      <div className="image-viewer">
        {previewStatus === "error" ? (
          <div className="preview-failure" role="alert">
            <strong>Preview unavailable</strong>
            <p>
              The image remains selected and can still be submitted for
              estimation.
            </p>
          </div>
        ) : (
          <figure className="image-stage">
            <img
              src={selectedImage.previewSource}
              alt={`Selected input: ${selectedImage.name}`}
              onLoad={(event) => {
                setDimensions({
                  width: event.currentTarget.naturalWidth,
                  height: event.currentTarget.naturalHeight,
                });
                setPreviewStatus("ready");
              }}
              onError={() => setPreviewStatus("error")}
            />

            {previewStatus === "ready" && result && (
              <PoseOverlay
                result={result}
                showSkeleton={overlayOptions.skeleton}
                showKeypoints={overlayOptions.keypoints}
                showBoxes={overlayOptions.boxes}
              />
            )}

            {result && (
              <div className="stage-badge">
                {backendLabel(result.backend)} ·{" "}
                {people.length === 0
                  ? "No people detected"
                  : `${formatPeopleCount(people.length)} detected`}
              </div>
            )}
          </figure>
        )}

        {previewStatus === "loading" && (
          <div className="preview-loading" role="status">
            Loading preview…
          </div>
        )}

        {estimateStatus === "processing" && (
          <div className="processing-overlay" role="status">
            <strong>Running pose estimation…</strong>
            <span>The selected image stays visible while processing.</span>
          </div>
        )}
      </div>

      {estimateError && (
        <div className="workspace-error" role="alert">
          <strong>Pose analysis error</strong>
          <p>{estimateError.message}</p>
        </div>
      )}

      {estimateStatus === "success" && people.length === 0 && (
        <div className="zero-result" role="status">
          No people detected.
        </div>
      )}
    </section>
  );
}
