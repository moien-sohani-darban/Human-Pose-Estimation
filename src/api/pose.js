import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  buildEstimateArguments,
  normalizeCommandError,
} from "./protocol.js";

const IMAGE_FILTER = {
  name: "Images",
  extensions: ["png", "jpg", "jpeg", "bmp", "webp"],
};

const MODEL_FILTERS = {
  mediapipe: { name: "MediaPipe model", extensions: ["task"] },
  yolo: { name: "YOLO Pose model", extensions: ["pt"] },
};

export async function getBackends() {
  try {
    return await invoke("python_get_backends");
  } catch (error) {
    throw normalizeCommandError(error);
  }
}

export async function estimatePose(options) {
  try {
    return await invoke(
      "python_estimate_pose",
      buildEstimateArguments(options),
    );
  } catch (error) {
    throw normalizeCommandError(error);
  }
}

export async function selectImageFile() {
  try {
    return await open({
      title: "Select an image",
      multiple: false,
      directory: false,
      filters: [IMAGE_FILTER],
    });
  } catch (error) {
    throw normalizeCommandError(error);
  }
}

export async function selectModelFile(backend) {
  const filter = MODEL_FILTERS[backend];
  if (!filter) return null;

  try {
    return await open({
      title: `Select ${filter.name}`,
      multiple: false,
      directory: false,
      filters: [filter],
    });
  } catch (error) {
    throw normalizeCommandError(error);
  }
}

export function createPreviewSource(path) {
  try {
    return convertFileSrc(path);
  } catch (error) {
    throw normalizeCommandError(error);
  }
}
