import { buildRenderablePeople } from "../utils/pose";

const percent = (value) => `${value * 100}%`;
const LIMB_COLORS = ["#7c55ff", "#22d3ee", "#34d399", "#fbbf24", "#fb7185", "#a78bfa"];

export default function PoseOverlay({
  result,
  showSkeleton,
  showKeypoints,
  showBoxes,
  skeletonStyle = "multi",
  lineThickness = 2,
  keypointSize = 7,
}) {
  const people = buildRenderablePeople(result);

  return (
    <svg
      className="pose-overlay"
      aria-hidden="true"
      focusable="false"
      preserveAspectRatio="none"
    >
      {people.map((person) => (
        <g key={`person-${person.personId}`}>
          {showBoxes && person.bbox && (
            <rect
              className="pose-box"
              x={percent(person.bbox.x)}
              y={percent(person.bbox.y)}
              width={percent(person.bbox.width)}
              height={percent(person.bbox.height)}
              stroke={person.color}
            />
          )}
          {showSkeleton &&
            person.segments.map(({ startName, endName, start, end }) => (
              <line
                className="pose-line-outline"
                key={`${person.personId}-${startName}-${endName}-outline`}
                x1={percent(start.x)}
                y1={percent(start.y)}
                x2={percent(end.x)}
                y2={percent(end.y)}
                strokeWidth={lineThickness + 2.5}
              />
            ))}
          {showSkeleton &&
            person.segments.map(({ startName, endName, start, end }, segmentIndex) => (
              <line
                className="pose-line"
                key={`${person.personId}-${startName}-${endName}`}
                x1={percent(start.x)}
                y1={percent(start.y)}
                x2={percent(end.x)}
                y2={percent(end.y)}
                stroke={skeletonStyle === "multi" ? LIMB_COLORS[segmentIndex % LIMB_COLORS.length] : person.color}
                strokeWidth={lineThickness}
              />
            ))}
          {showKeypoints &&
            person.keypoints.map((point) => (
              <g key={`${person.personId}-point-${point.index}-${point.name}`}>
                <circle
                  className="pose-point-outline"
                  cx={percent(point.x)}
                  cy={percent(point.y)}
                  r={keypointSize / 2 + 1.8}
                />
                <circle
                  className="pose-point"
                  cx={percent(point.x)}
                  cy={percent(point.y)}
                  r={keypointSize / 2}
                  fill={person.color}
                />
              </g>
            ))}
        </g>
      ))}
    </svg>
  );
}
