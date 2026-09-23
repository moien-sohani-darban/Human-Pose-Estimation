import { useEffect, useRef, useState } from "react";
import Icon from "./Icon";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 2;
const ZOOM_STEP = 0.1;

const MediaViewer = ({
  children,
  className = "",
  toolbar = true
}) => {
  const frameRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const document = globalThis.document;
    if (!document?.addEventListener) return undefined;

    const syncFullscreenState = () => {
      setIsFullscreen(document.fullscreenElement === frameRef.current);
    };

    document.addEventListener("fullscreenchange", syncFullscreenState);
    syncFullscreenState();
    return () => document.removeEventListener("fullscreenchange", syncFullscreenState);
  }, []);

  function changeZoom(delta) {
    setZoom((current) =>
      Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number((current + delta).toFixed(1)))),
    );
  }

  async function toggleFullscreen() {
    const document = globalThis.document;
    if (document?.fullscreenElement === frameRef.current) {
      await document.exitFullscreen?.();
    } else {
      await frameRef.current?.requestFullscreen?.();
    }
  }

  return (
    <div className={`media-viewer ${className}`.trim()} ref={frameRef}>
      <div className="media-viewport">
        <div className="media-zoom-layer" style={{ transform: `scale(${zoom})` }}>
          {children}
        </div>
      </div>

      {toolbar && (
        <div className="media-toolbar">
          <button
            className="media-toolbar-button"
            type="button"
            onClick={() => changeZoom(-ZOOM_STEP)}
            disabled={zoom <= MIN_ZOOM}
          >
            <Icon name="zoomOut" size={14} />
            <span>zoom out</span>
          </button>

          <span className="media-toolbar-output">{Math.round(zoom * 100)}%</span>

          <button
            className="media-toolbar-button"
            type="button"
            onClick={() => changeZoom(ZOOM_STEP)}
            disabled={zoom >= MAX_ZOOM}
          >
            <Icon name="zoomIn" size={14} />
            <span>zoom in</span>
          </button>

          <button
            className="media-toolbar-button"
            type="button"
            onClick={() => setZoom(1)}>
            Fit to view
          </button>

          <button
            className="media-toolbar-button"
            type="button"
            onClick={() => setZoom(1)}
          >
            Reset view
          </button>

          <button
            className="media-toolbar-button"
            type="button"
            onClick={toggleFullscreen}
            disabled={!isFullscreen && !globalThis.document?.fullscreenEnabled}
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            <Icon name={isFullscreen ? "fullscreenExit" : "fullscreen"} size={14} />
            <span>{isFullscreen ? "Exit Fullscreen" : "Fullscreen"}</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default MediaViewer;
