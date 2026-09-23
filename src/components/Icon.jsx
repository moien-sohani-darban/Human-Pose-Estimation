const browseDark = new URL("../assets/icons/BrowseDark.svg", import.meta.url).href;
const browseLight = new URL("../assets/icons/BrowseLight.svg", import.meta.url).href;
const camera = new URL("../assets/icons/Camera.svg", import.meta.url).href;
const cameraOff = new URL("../assets/icons/CameraOff.svg", import.meta.url).href;
const caution = new URL("../assets/icons/Caution.svg", import.meta.url).href;
const chevron = new URL("../assets/icons/Chevron.svg", import.meta.url).href;
const close = new URL("../assets/icons/Close.svg", import.meta.url).href;
const exitFullscreen = new URL("../assets/icons/ExitFullScreen.svg", import.meta.url).href;
const fullscreen = new URL("../assets/icons/FullScreen.svg", import.meta.url).href;
const image = new URL("../assets/icons/Image.svg", import.meta.url).href;
const model = new URL("../assets/icons/Model.svg", import.meta.url).href;
const play = new URL("../assets/icons/Play.svg", import.meta.url).href;
const preferences = new URL("../assets/icons/Preferences.svg", import.meta.url).href;
const result = new URL("../assets/icons/Result.svg", import.meta.url).href;
const resume = new URL("../assets/icons/Resume.svg", import.meta.url).href;
const speaker = new URL("../assets/icons/Speaker.svg", import.meta.url).href;
const stop = new URL("../assets/icons/Stop.svg", import.meta.url).href;
const uploadImage = new URL("../assets/icons/UploadImage.svg", import.meta.url).href;
const uploadVideo = new URL("../assets/icons/UploadVideo.svg", import.meta.url).href;
const video = new URL("../assets/icons/Video.svg", import.meta.url).href;
const visual = new URL("../assets/icons/Visual.svg", import.meta.url).href;
const webcam = new URL("../assets/icons/Webcam.svg", import.meta.url).href;
const zoomIn = new URL("../assets/icons/ZoomIn.svg", import.meta.url).href;
const zoomOut = new URL("../assets/icons/ZoomOut.svg", import.meta.url).href;

const assets = {
  camera,
  cameraOff,
  chevron,
  close,
  fullscreen,
  fullscreenExit: exitFullscreen,
  image,
  model,
  play,
  preferences,
  results: result,
  resume,
  speaker,
  stop,
  uploadImage,
  uploadVideo,
  video,
  visual,
  warning: caution,
  webcam,
  zoomIn,
  zoomOut,
};

const adaptiveIcons = new Set([
  "camera",
  "cameraOff",
  "fullscreen",
  "fullscreenExit",
  "image",
  "model",
  "results",
  "stop",
  "uploadImage",
  "uploadVideo",
  "video",
  "visual",
  "webcam",
]);

const fallbackPaths = {
  refresh: <><path d="M20 7v5h-5" /><path d="M19 12a7 7 0 1 0-2 5" /></>,
};

export default function Icon({ name, size = 18, className = "" }) {
  if (name === "folder") {
    return (
      <span className={`icon themed-icon ${className}`.trim()} style={{ width: size, height: size }} aria-hidden="true">
        <img className="themed-icon-light" src={browseLight} alt="" />
        <img className="themed-icon-dark" src={browseDark} alt="" />
      </span>
    );
  }

  const source = assets[name];
  if (source) {
    if (adaptiveIcons.has(name)) {
      return (
        <span
          className={`icon asset-icon theme-adaptive-icon ${className}`.trim()}
          style={{ "--icon-source": `url("${source}")`, width: size, height: size }}
          aria-hidden="true"
        />
      );
    }

    return (
      <img
        className={`icon asset-icon ${className}`.trim()}
        src={source}
        width={size}
        height={size}
        alt=""
        aria-hidden="true"
      />
    );
  }

  return (
    <svg
      className={`icon ${className}`.trim()}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {fallbackPaths[name] ?? fallbackPaths.refresh}
    </svg>
  );
}
