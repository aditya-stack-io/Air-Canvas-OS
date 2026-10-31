# Air-Canvas OS: Gesture & Voice-Controlled Spatial Desktop Interface

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

**Air-Canvas OS** transforms any standard webcam and microphone into a touchless spatial desktop environment. Sketch in mid-air with dynamic brush sizing, control your OS cursor with exponential smoothing, scroll through documents, and trigger workspace commands hands-free using background voice recognition.

---

## Core Features

1. **Multi-Mode Spatial Engine (`CANVAS`, `CURSOR`, `SCROLL`)**
   - **Canvas Mode:** Low-latency freehand air-drawing, interactive top HUD dock selection, dynamic pinch-to-resize brush, 15-step Undo/Redo history stack, and one-click PNG artwork export.
   - **Cursor Mode:** Maps 21-point 3D hand landmarks to full-screen OS coordinates with exponential smoothing, Frame Margin dead-zone compensation, Left-Click pinch (`Thumb + Index`), and Right-Click pinch (`Thumb + Middle`).
   - **Scroll Mode:** Touchless two-finger vertical scrolling with proportional speed control.

2. **Asynchronous Voice Command HUD**
   - Runs `SpeechRecognition` and `pyttsx3` TTS feedback on non-blocking daemon threads so camera FPS never drops.
   - Gracefully falls back to HUD & keyboard controls if no microphone is plugged in.

3. **Zero-Dependency Fallback Controls**
   - Full interactive top bar + keyboard hotkeys (`M`, `C`, `U`, `R`, `S`, `+`, `-`, `Q`).

---

## Gesture, Voice & Keyboard Control Matrix

| Mode / Layer | Input / Trigger | Action |
| :--- | :--- | :--- |
| **Top Spatial Dock** | Index + Middle Finger Hover (`y < 78px`) | Select Color, Undo, Redo, Clear, Save, or Switch Mode |
| **Canvas Mode** | Index Finger Up Only | Freehand draw / erase on spatial canvas |
| **Canvas Mode** | Index + Middle Finger Up | Hover pointer without drawing |
| **Canvas Mode** | Thumb + Index Pinch Distance | Dynamically resize brush (`2px` – `45px`) |
| **Cursor Mode** | Index Finger Up | Move OS mouse cursor smoothly |
| **Cursor Mode** | Thumb + Index Pinch | Left Mouse Click |
| **Cursor Mode** | Thumb + Middle Pinch | Right Mouse Click |
| **Scroll Mode** | Index + Middle Finger Up (Move Up/Down) | Vertical OS Scroll Up / Down |
| **Voice Command** | *"Canvas mode"* / *"Cursor mode"* / *"Scroll mode"* | Switch active spatial operating mode |
| **Voice Command** | *"Color [Cyan / Magenta / Green / Amber / White]"* | Switch active brush color |
| **Voice Command** | *"Eraser"* / *"Undo"* / *"Redo"* / *"Clear"* / *"Save"* | Trigger workspace tool or export PNG |
| **Keyboard** | `M` / `C` / `U` / `R` / `S` / `Q` | Mode / Clear / Undo / Redo / Save / Quit |

---

## Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/yourusername/air-canvas-os.git
cd air-canvas-os
pip install -r requirements.txt
```

### 2. Run the Standalone All-in-One File
```bash
python air_canvas_os.py
```
*(Or run the modular entry point via `python src/main.py`)*
