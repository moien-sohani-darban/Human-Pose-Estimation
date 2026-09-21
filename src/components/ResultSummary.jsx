import {
  backendLabel,
  formatDimensions,
  formatPeopleCount,
  formatProcessingTime,
} from "../utils/pose";

export default function ResultSummary({ result }) {
  if (!result || !Array.isArray(result.people)) {
    return (
      <section className="result-summary">
        <h2>Result</h2>
        <p className="empty-results">
          Run pose estimation to see detection metadata.
        </p>
      </section>
    );
  }

  return (
    <section className="result-summary">
      <h2>Detection summary</h2>

      <div className="summary-grid">
        <div>
          <span>Backend</span>
          <strong>{backendLabel(result.backend)}</strong>
        </div>

        <div>
          <span>People detected</span>
          <strong>{formatPeopleCount(result.people.length)}</strong>
        </div>

        <div>
          <span>Processing time</span>
          <strong>{formatProcessingTime(result.processing_time_ms)}</strong>
        </div>

        <div>
          <span>Image dimensions</span>
          <strong>
            {formatDimensions(result.image_width, result.image_height)}
          </strong>
        </div>
      </div>
    </section>
  );
}
