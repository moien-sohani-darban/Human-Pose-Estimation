import { useEffect, useState } from "react";
import { backendLabel, formatPeopleCount } from "../utils/pose";
import Icon from "./Icon";
import MediaViewer from "./MediaViewer";
import PoseOverlay from "./PoseOverlay";

const ImageWorkspace = ({
  selectedImage,
  result,
  estimateStatus,
  estimateError,
  overlayOptions,
  onChooseImage,
}) => {
  const [previewStatus, setPreviewStatus] = useState("loading");
  const [dimensions, setDimensions] = useState(null);

  useEffect(() => {
    setPreviewStatus(selectedImage ? "loading" : "empty");
    setDimensions(null);
  }, [selectedImage]);

  const people = Array.isArray(result?.people) ? result.people : [];
  const keypointCount = people.reduce(
    (total, person) => total + (Array.isArray(person.keypoints) ? person.keypoints.length : 0),
    0,
  );

  return (
    <div className="mode-workspace">
      <header className="workspace-heading">
        <div className="workspace-heading-contents">
          <span className="workspace-heading-title">Image Analysis</span>
          <span className="workspace-heading-subtitle">
            {selectedImage ?
              `${selectedImage.name}${dimensions ?
                ` · ${dimensions.width} × ${dimensions.height}` :
                ""}` :
              "Select an image to begin"
            }
          </span>
        </div>

        {estimateStatus === "success" &&
          <span className="status-chip success">
            <i />
            Analysis complete
          </span>}
      </header>

      {!selectedImage ? (
        <div className="media-stage empty-stage">
          <div className="empty-visual">
            <Icon name="uploadImage" size={36} />
          </div>

          <div className="empty-stage-contents">
            <span className="empty-stage-title">Choose an image to begin</span>
            <span className="empty-stage-subtitle">Select a local image and run pose estimation</span>
          </div>

          <button
            className="primary-button compact-button"
            type="button"
            onClick={onChooseImage}
          >
            <Icon name="uploadImage" size={16} />
            Choose Image
          </button>
        </div>
      ) : (
        <>
          <MediaViewer className="image-viewer">
            {previewStatus === "error" ? (
              <div className="preview-failure" role="alert"><strong>Preview unavailable</strong><p>The image remains selected and can still be submitted for estimation.</p></div>
            ) : (
              <figure className="image-stage media-content-stage">
                <img
                  src={selectedImage.previewSource}
                  alt={`Selected input: ${selectedImage.name}`}
                  onLoad={(event) => {
                    setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight });
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
                    skeletonStyle={overlayOptions.skeletonStyle}
                    lineThickness={overlayOptions.lineThickness}
                    keypointSize={overlayOptions.keypointSize}
                  />
                )}
                {result && (
                  <div className="stage-badges">
                    <span className="stage-badge">{backendLabel(result.backend)} · {people.length === 0 ? "No people detected" : `${formatPeopleCount(people.length)} detected · ${keypointCount} keypoints`}</span>
                  </div>
                )}
              </figure>
            )}
            {previewStatus === "loading" && <div className="preview-loading" role="status">Loading preview…</div>}
            {estimateStatus === "processing" && <div className="processing-overlay" role="status"><span className="spinner" /><strong>Running pose estimation…</strong><span>The selected image stays visible while the local engine works.</span></div>}
          </MediaViewer>
          {estimateError && <div className="workspace-error" role="alert"><strong>Pose analysis error</strong><p>{estimateError.message}</p></div>}
          {estimateStatus === "success" && people.length === 0 && <div className="zero-result" role="status">No people detected.</div>}
        </>
      )}
    </div>
  );
};

export default ImageWorkspace;
