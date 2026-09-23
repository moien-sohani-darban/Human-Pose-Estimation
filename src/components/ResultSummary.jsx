import { useState } from "react";
import {
  backendLabel,
  formatPeopleCount,
  formatProcessingTime,
} from "../utils/pose";
import Icon from "./Icon";

function PersonDetail({ person, index, initiallyOpen }) {
  const [open, setOpen] = useState(initiallyOpen);
  const keypointCount = Array.isArray(person.keypoints) ? person.keypoints.length : 0;

  return (
    <button
      type="button"
      className={`person-detail ${open ? "open" : ""}`}
      aria-expanded={open}
      onClick={() => setOpen((current) => !current)}
    >
      <span className="person-detail-header">
        Person {index + 1}
        <Icon name="chevron" className="person-detail-chevron" size={8} />
      </span>

      <span className="person-detail-info">
        <span className="person-detail-info-text">Keypoints: {keypointCount}</span>
        <span className="person-detail-info-text">Bounding Box: {person.bbox ? "Available" : "Unavailable"}</span>
      </span>
    </button>
  );
}

export default function ResultSummary({ result, inferenceFps = null, mode = "image" }) {
  const people = Array.isArray(result?.people) ? result.people : [];
  const emptyMessage = mode === "image"
    ? "No results yet — Run inference to see detection results."
    : mode === "webcam"
      ? "No live results yet — Start the camera and live estimation."
      : "No video results yet — Play the video and start pose analysis.";

  return (
    <div className="inspector-card results-card">
      <header className="card-heading">
        <div className="card-heading-left">
          <Icon name="results" />
          <span className="card-heading-title">Results</span>
        </div>
      </header>

      {!result || !Array.isArray(result.people) ? (
        <span className="empty-results">{emptyMessage}</span>
      ) : (
        <div className="result-contents">
          <div className="result-overview">
            <span className="result-overview-title">{formatPeopleCount(people.length)}</span>
            <span className="result-overview-timer">{formatProcessingTime(result.processing_time_ms)}</span>
          </div>

          <div className="person-results">
            {people.map((person, index) => (
              <PersonDetail
                key={person.person_id ?? index}
                person={person}
                index={index}
                initiallyOpen={index === 0}
              />
            ))}
          </div>

          <div className="field-row">
            <span className="field-row-label">Backend</span>
            <span className="field-row-value">{backendLabel(result.backend)}</span>
          </div>

          <div className="field-row">
            <span className="field-row-label">People</span>
            <span className="field-row-value">{people.length}</span>
          </div>

          <div className="field-row">
            <span className="field-row-label">Processing Time</span>
            <span className="field-row-value">{formatProcessingTime(result.processing_time_ms)}</span>
          </div>

          {mode !== "image" && (
            <div className="field-row">
              <span className="field-row-label">Inference FPS</span>
              <span className="field-row-value">{Number.isFinite(inferenceFps) ? inferenceFps.toFixed(1) : "—"}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
