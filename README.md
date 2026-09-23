<div align="center">

# 🧍 Human Pose Estimation

### A multi-backend desktop application for human pose estimation across images, webcam streams, and video files

A local-first computer vision application built with **React**, **Tauri**, **Rust**, and **Python**, supporting both **MediaPipe Pose Landmarker** and **YOLO Pose** through a unified pose-estimation engine.

<br>

[![Python](https://img.shields.io/badge/Python-3.13.5-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Tauri](https://img.shields.io/badge/Tauri-2-24C8DB?style=for-the-badge&logo=tauri&logoColor=white)](https://tauri.app/)
[![Rust](https://img.shields.io/badge/Rust-Desktop%20Bridge-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-0097A7?style=for-the-badge)](https://ai.google.dev/edge/mediapipe/)
[![YOLO](https://img.shields.io/badge/YOLO-Pose-111111?style=for-the-badge)](https://docs.ultralytics.com/tasks/pose/)

<br>

[Overview](#-overview) •
[Features](#-features) •
[Architecture](#-architecture) •
[Backends](#-supported-backends) •
[Installation](#-installation) •
[Development](#-development) •
[Testing](#-testing)

</div>

---

## 📌 Overview

**Human Pose Estimation** is a local Windows desktop application for estimating human body poses from:

- Still images
- Live webcam input
- Local video files

The application combines a modern React interface with a Tauri/Rust desktop layer and a persistent Python computer-vision engine.

Instead of coupling the interface directly to one pose-estimation library, the project uses a backend-independent pose contract that allows different inference engines to expose results through one normalized representation.

Currently supported pose backends include:

- **MediaPipe Pose Landmarker**
- **Ultralytics YOLO Pose**

All inference runs locally on the user's machine. The application does not rely on a web server or cloud inference service.

---

## ✨ Features

- 🖼️ Human pose estimation from still images
- 📷 Live webcam pose estimation
- 🎬 Pose analysis for local video files
- 🧠 Multiple pose-estimation backends
- 🧍 Multi-person pose support
- 🦴 Skeleton and keypoint visualization
- 📦 Optional person bounding boxes
- 🔄 Backend-independent pose representation
- ⚡ Persistent Python inference process
- 🔐 Typed Rust ↔ Python communication
- 📴 Offline-friendly runtime behavior
- 📂 Custom local model selection
- 🧩 Explicit model-asset handling
- 🪟 Native Windows desktop packaging
- 🧪 Python, frontend, and Rust test suites

---

## 🎬 Application Modes

### Image Mode

Load a supported image and estimate one or more human poses.

The interface can display:

- keypoints
- skeleton connections
- bounding boxes
- detected person count
- image dimensions
- inference backend
- processing time

### Webcam Mode

The application supports live webcam estimation using bounded frame processing.

Only one inference request is kept active at a time, preventing stale frames from building up when inference is slower than the webcam preview.

### Video Mode

Local videos can be played using native video controls while pose analysis samples fresh frames during playback.

Old results are discarded after seeking or when they fall too far behind the current playback position.

---

## 🧠 How It Works

The system separates the desktop interface from the computer-vision engine.

```text
┌───────────────────────────────┐
│          Input Source         │
│                               │
│   Image   Webcam   Video      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│          React UI             │
│                               │
│ Preview • Controls • Overlay  │
└───────────────┬───────────────┘
                │
         Tauri Commands
                │
                ▼
┌───────────────────────────────┐
│        Rust / Tauri           │
│                               │
│   Sidecar Process Manager     │
│ Request IDs • Timeouts        │
└───────────────┬───────────────┘
                │
          NDJSON IPC
                │
                ▼
┌───────────────────────────────┐
│      Python Sidecar           │
│                               │
│         PoseEngine            │
└───────────┬─────────┬─────────┘
            │         │
            ▼         ▼
      MediaPipe      YOLO Pose
            │         │
            └────┬────┘
                 ▼
        Unified PoseResult
                 │
                 ▼
          SVG Pose Overlay
```

---

## 🏗️ Architecture

The project uses a persistent sidecar architecture rather than starting a Python process for every inference request.

This design provides several advantages:

- model instances can be reused
- repeated Python startup overhead is avoided
- communication remains explicit and testable
- frontend code stays independent of backend-specific inference objects
- long-running webcam and video analysis remains memory-bounded

Communication between Rust and Python uses line-delimited JSON over standard process streams.

---

## 🔌 Supported Backends

| Backend | Status | Output |
| :--- | :---: | :--- |
| **MediaPipe Pose Landmarker** | ✅ Supported | Up to 33 canonical landmarks |
| **YOLO Pose** | ✅ Supported | COCO 17-keypoint pose observations, including multi-person output |
| **MMPose** | ⚠️ Unavailable | Current Python 3.13 / Windows dependency stack is not reproducible |

Both supported backends are normalized into the same application-level pose contract.

This means the React interface and Rust bridge do not need backend-specific rendering logic.

---

## 🧩 Unified Pose Representation

One of the key design decisions in this project is the use of a backend-independent pose model.

Instead of exposing raw MediaPipe or Ultralytics objects to the application, each backend is adapted into a common `PoseResult` structure.

Conceptually:

```text
MediaPipe Result ─┐
                  │
                  ├──► PoseResult ───► Application
                  │
YOLO Pose Result ─┘
```

This makes backend selection transparent to the rest of the application.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Interface** | React 19, Vite 7, JavaScript |
| **Desktop Shell** | Tauri 2 |
| **Native Bridge** | Rust |
| **Vision Engine** | Python 3.13.5 |
| **Image Processing** | OpenCV, NumPy |
| **Pose Backend** | MediaPipe Tasks |
| **Pose Backend** | Ultralytics YOLO Pose |
| **IPC** | Persistent NDJSON over process streams |
| **Packaging** | PyInstaller, Tauri, NSIS |
| **Target Platform** | Windows x64 |

---

## 📁 Project Structure

```text
Human-Pose-Estimation/
│
├── src/
│   └── React user interface, overlays, and media schedulers
│
├── src-tauri/
│   └── Rust bridge, Tauri commands, sidecar management, and packaging
│
├── python-engine/
│   └── Pose backends, PoseEngine, protocol, serialization, and tests
│
├── scripts/
│   └── Development and Windows release automation
│
├── index.html
├── package.json
├── vite.config.js
├── yarn.lock
├── .gitignore
└── README.md
```

---

## 📦 Model Setup

Model weights are intentionally kept external to the repository.

The application does not silently download pose models.

### MediaPipe

Default development location:

```text
python-engine/models/mediapipe/pose_landmarker.task
```

### YOLO Pose

Default development location:

```text
python-engine/models/yolo/yolo11n-pose.pt
```

Users can also select compatible local model files through the application interface.

---

## 🚀 Installation

### Windows Application

The packaged application targets:

```text
Windows 11 x64
```

The release build includes the Python runtime and required Python dependencies.

End users therefore do **not** need to separately install:

- Python
- pip
- Node.js
- Rust
- Tauri

The current installer is generated using NSIS.

> The installer is currently unsigned, so Windows SmartScreen may display a warning.

---

## 💻 Development

### Requirements

For source development:

- Windows 11 x64
- Node.js
- Yarn 1.x
- Rust toolchain
- Tauri Windows prerequisites
- Python 3.13.5

### Python Environment

From the repository root:

```powershell
python -m venv python-engine\.venv
```

Install Python dependencies:

```powershell
.\python-engine\.venv\Scripts\python.exe -m pip install --upgrade pip
.\python-engine\.venv\Scripts\python.exe -m pip install -e ".\python-engine[test]"
```

Install frontend dependencies:

```powershell
yarn install
```

Run the desktop application in development mode:

```powershell
yarn tauri dev
```

---

## 🏭 Production Build

Install the packaging dependencies:

```powershell
.\python-engine\.venv\Scripts\python.exe -m pip install -e ".\python-engine[package]"
```

Run the Windows release build:

```powershell
.\scripts\build-windows-release.ps1
```

The build process:

1. verifies the expected Python runtime
2. packages the Python sidecar
3. smoke-tests the packaged sidecar
4. embeds the runtime into the Tauri application
5. builds the frontend
6. builds the Rust desktop shell
7. creates an NSIS Windows installer

Generated installers are placed under:

```text
src-tauri/target/release/bundle/nsis/
```

---

## 🧪 Testing

The repository contains separate test suites for Python, frontend, and Rust components.

### Frontend

```bash
yarn test
yarn build
```

### Python

```powershell
cd python-engine

.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m integration -ra
```

### Rust / Tauri

```powershell
cd src-tauri

cargo fmt --check
cargo check --locked
cargo test --locked
cargo clippy --locked -- -D warnings
```

The latest verified project state includes:

```text
243 Python tests passed
43 frontend tests passed
23 active Rust tests passed
```

The packaged Python sidecar smoke test and packaged-runtime protocol checks also pass.

---

## ⚙️ Engineering Highlights

Some of the main engineering decisions behind the project include:

### Persistent Python Sidecar

Python remains alive between requests, allowing initialized pose estimators to be reused.

### Backend Abstraction

MediaPipe and YOLO Pose are hidden behind one application-level estimator contract.

### Typed Error Propagation

Structured error codes are preserved across:

```text
Python
  ↓
NDJSON
  ↓
Rust
  ↓
React
```

while the UI presents safe, readable guidance to users.

### Bounded Live Processing

Webcam and video analysis keep at most one inference request active.

This prevents unbounded frame queues and stale inference results.

### Explicit Model Assets

Model downloads never happen implicitly.

This keeps the application deterministic and makes offline usage possible.

---

## ⚠️ Current Limitations

- Windows x64 is currently the only packaged target
- The Windows installer is unsigned
- Production packaging currently uses CPU-only Torch
- Model files must be provided separately
- MMPose is currently unavailable in the supported Python/Windows environment
- Temporal tracking and pose smoothing are not implemented
- Action recognition is not implemented
- Processed video export is not currently supported

---

## 🗺️ Roadmap

Potential future improvements include:

- [ ] Project-specific application icon
- [ ] Signed Windows release
- [ ] Public downloadable installer
- [ ] GPU-enabled production package
- [ ] Cross-platform desktop builds
- [ ] Temporal pose smoothing
- [ ] Multi-frame person tracking
- [ ] Action recognition
- [ ] Processed video export
- [ ] Additional pose-estimation backends
- [ ] Revisit MMPose when dependency support improves

---

## 📸 Screenshots

Real application screenshots should be added to:

```text
docs/images/image-mode.png
docs/images/webcam-mode.png
docs/images/video-mode.png
```

Recommended README layout:

```markdown
### Image Mode

![Image Mode](docs/images/image-mode.png)

### Webcam Mode

![Webcam Mode](docs/images/webcam-mode.png)

### Video Mode

![Video Mode](docs/images/video-mode.png)
```

Using real release screenshots is preferable to using mockups because it shows the actual application state.

---

## 🎯 Project Purpose

This project explores both the computer-vision and software-engineering aspects of deploying pose-estimation models in a desktop application.

It demonstrates:

- human pose estimation
- multi-backend computer vision
- model abstraction
- local process communication
- desktop application architecture
- live-media processing
- Rust/Python integration
- React visualization
- Windows application packaging
- automated testing across multiple technology stacks

---

## 🤝 Contributing

Contributions, suggestions, and issue reports are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/your-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push the branch

```bash
git push origin feature/your-feature
```

5. Open a Pull Request

---

## 👤 Author

<div align="center">

### Moien Sohani Darban

[![GitHub](https://img.shields.io/badge/GitHub-moiensohani-181717?style=for-the-badge&logo=github)](https://github.com/moien-sohani-darban)

</div>

---

<div align="center">

### ⭐ If you find this project useful, consider giving it a star.

**Built with React, Tauri, Rust, Python, MediaPipe, and YOLO Pose**

</div>
