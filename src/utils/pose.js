// Mirrors python-engine/app/utils/visualization.py CANONICAL_SKELETON.
export const CANONICAL_SKELETON = Object.freeze([
  ["nose", "left_eye_inner"],
  ["left_eye_inner", "left_eye"],
  ["left_eye", "left_eye_outer"],
  ["left_eye_outer", "left_ear"],
  ["nose", "right_eye_inner"],
  ["right_eye_inner", "right_eye"],
  ["right_eye", "right_eye_outer"],
  ["right_eye_outer", "right_ear"],
  ["mouth_left", "mouth_right"],
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["left_wrist", "left_pinky"],
  ["left_wrist", "left_index"],
  ["left_wrist", "left_thumb"],
  ["left_pinky", "left_index"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["right_wrist", "right_pinky"],
  ["right_wrist", "right_index"],
  ["right_wrist", "right_thumb"],
  ["right_pinky", "right_index"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["left_ankle", "left_heel"],
  ["left_heel", "left_foot_index"],
  ["left_ankle", "left_foot_index"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["right_ankle", "right_heel"],
  ["right_heel", "right_foot_index"],
  ["right_ankle", "right_foot_index"],
]);

export const PERSON_COLORS = Object.freeze([
  "#38bdf8",
  "#fb7185",
  "#a3e635",
  "#fbbf24",
  "#c084fc",
  "#2dd4bf",
]);

export function isRenderablePoint(point) {
  return (
    point !== null &&
    typeof point === "object" &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y)
  );
}

export function createKeypointLookup(person) {
  const byName = new Map();
  const byIndex = new Map();
  const keypoints = Array.isArray(person?.keypoints) ? person.keypoints : [];

  for (const point of keypoints) {
    if (!isRenderablePoint(point)) continue;
    if (typeof point.name === "string") byName.set(point.name, point);
    if (Number.isInteger(point.index)) byIndex.set(point.index, point);
  }

  return { byName, byIndex };
}

export function resolveSkeletonSegments(person) {
  const { byName } = createKeypointLookup(person);
  return CANONICAL_SKELETON.flatMap(([startName, endName]) => {
    const start = byName.get(startName);
    const end = byName.get(endName);
    return start && end ? [{ startName, endName, start, end }] : [];
  });
}

function safeBoundingBox(bbox) {
  if (
    bbox &&
    [bbox.x, bbox.y, bbox.width, bbox.height].every(Number.isFinite)
  ) {
    return bbox;
  }
  return null;
}

export function buildRenderablePeople(result) {
  const people = Array.isArray(result?.people) ? result.people : [];
  return people.map((person, order) => {
    const keypoints = Array.isArray(person?.keypoints)
      ? person.keypoints.filter(isRenderablePoint)
      : [];
    const numericId = Number.isInteger(person?.person_id)
      ? person.person_id
      : order;
    const colorIndex = ((numericId % PERSON_COLORS.length) + PERSON_COLORS.length) % PERSON_COLORS.length;
    return {
      personId: numericId,
      color: PERSON_COLORS[colorIndex],
      keypoints,
      segments: resolveSkeletonSegments({ keypoints }),
      bbox: safeBoundingBox(person?.bbox),
    };
  });
}

export function validatePoseResult(result) {
  return Boolean(
    result &&
      typeof result === "object" &&
      typeof result.success === "boolean" &&
      typeof result.backend === "string" &&
      Array.isArray(result.people),
  );
}

export function normalizeModelMetadata(models) {
  return models && typeof models === "object" && !Array.isArray(models)
    ? models
    : {};
}

export function getModelStatus(metadata, customPath) {
  if (typeof customPath === "string" && customPath.trim()) {
    return {
      kind: "custom",
      label: "Custom model selected",
      guidance: "The selected file will be validated when the backend initializes.",
    };
  }
  if (metadata?.exists === true) {
    return {
      kind: "ready",
      label: "Default model ready",
      guidance: "The canonical local model file is present.",
    };
  }
  if (metadata?.exists === false) {
    return {
      kind: "missing",
      label: "Default model missing",
      guidance:
        "Select a compatible local model with Browse. The expected default is shown below.",
    };
  }
  return {
    kind: "unknown",
    label: "Model unavailable",
    guidance: "Refresh backend availability to check the default model.",
  };
}

export function withModelSetupGuidance(error, backend, defaultPath = "") {
  if (error?.code !== "model_asset_not_found") return error;
  const backendName = backend ? backendLabel(backend) : "This backend";
  const expected = defaultPath ? ` Expected default: ${defaultPath}.` : "";
  return {
    ...error,
    message: `${backendName} needs a compatible local model. Select one with Browse.${expected}`,
  };
}

export function getDefaultModelDisplayPath(metadata) {
  const value = metadata?.display_path;
  if (typeof value !== "string" || !value.trim()) return "";
  const clean = value.trim().replaceAll("\\", "/");
  if (clean.startsWith("/") || /^[a-zA-Z]:\//.test(clean)) return "";
  return clean;
}

export function setBackendModelPath(modelPaths, backend, path) {
  return { ...modelPaths, [backend]: path };
}

export function fileNameFromPath(path) {
  if (typeof path !== "string") return "Selected image";
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? "Selected image";
}

export function backendLabel(backend) {
  return (
    { mediapipe: "MediaPipe", yolo: "YOLO Pose", mmpose: "MMPose" }[
      backend
    ] ?? backend
  );
}

export function formatPeopleCount(count) {
  return `${count} ${count === 1 ? "person" : "people"}`;
}

export function formatProcessingTime(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} ms` : "Not reported";
}

export function formatDimensions(width, height) {
  return Number.isFinite(width) && Number.isFinite(height)
    ? `${width} × ${height}`
    : "Not reported";
}
