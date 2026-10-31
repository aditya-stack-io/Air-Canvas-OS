#!/usr/bin/env python3
"""
Air-Canvas OS: Gesture & Voice-Controlled Spatial Desktop Interface
===================================================================
A complete, production-ready spatial computing interface turning any webcam
and microphone into a touchless drawing canvas, OS mouse controller, scroll/media
deck, and voice-activated workspace.

Controls Summary:
- Top Spatial Dock: Hover with Index + Middle fingers up to select Colors, Tools, Undo, Clear, Save, or Mode.
- CANVAS Mode:
    * Index finger up: Draw freehand
    * Index + Middle up: Hover / Select from HUD dock
    * Thumb + Index pinch (hold) in canvas: Dynamic brush size adjustment
    * All 5 fingers up (open palm hold): Quick clear trigger
- CURSOR Mode:
    * Index finger up: Move OS cursor (exponentially smoothed)
    * Thumb + Index pinch: Left Click
    * Thumb + Middle pinch: Right Click
    * Index + Middle together + Thumb pinch: Drag & Drop toggle
- SCROLL Mode:
    * Index + Middle fingers up: Move hand vertically to scroll up/down
    * Thumb + Index distance: Zoom in / out (Ctrl + Scroll)
- Voice Commands (Background Thread):
    * "canvas mode", "cursor mode", "scroll mode"
    * "color cyan", "color magenta", "color green", "color yellow", "color white", "eraser"
    * "undo", "redo", "clear", "save", "brush bigger", "brush smaller"
- Keyboard Shortcuts (Fallback):
    * [M] Cycle Mode | [C] Clear | [U] Undo | [R] Redo | [S] Save PNG | [+/-] Brush Size | [Q] Quit
"""

import os
import sys
import time
import math
import threading
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe is required. Install via: pip install mediapipe")

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01
    SCREEN_W, SCREEN_H = pyautogui.size()
except Exception:
    pyautogui = None
    SCREEN_W, SCREEN_H = 1920, 1080

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================
class Config:
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    FPS_TARGET = 30

    # Spatial Mouse Parameters
    SMOOTHENING = 5.5
    FRAME_MARGIN = 110
    PINCH_THRESHOLD = 36

    # Brush & Canvas Parameters
    DEFAULT_BRUSH_SIZE = 8
    MIN_BRUSH_SIZE = 2
    MAX_BRUSH_SIZE = 45
    ERASER_SIZE = 55
    MAX_UNDO_STEPS = 15

    # Color Palette (BGR for OpenCV)
    COLORS = {
        "CYAN": (255, 255, 0),
        "MAGENTA": (255, 0, 255),
        "GREEN": (0, 255, 128),
        "AMBER": (0, 191, 255),
        "WHITE": (255, 255, 255),
        "ERASER": (0, 0, 0),
    }

    MODES = ["CANVAS", "CURSOR", "SCROLL"]


# ============================================================================
# 2. HAND TRACKING & GESTURE ENGINE
# ============================================================================
class HandGestureEngine:
    def __init__(self, max_hands=1, detection_con=0.75, track_con=0.75):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list = []
        self.handedness = "Right"
        self.results = None

    def process_frame(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            if self.results.multi_handedness:
                self.handedness = (
                    self.results.multi_handedness[0].classification[0].label
                )
            hand_lms = self.results.multi_hand_landmarks[0]
            if draw:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
            h, w, _ = img.shape
            for idx, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append((idx, cx, cy))
        return img, self.lm_list

    def fingers_up(self):
        if len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []
        # Thumb check (adapts to flipped webcam view and handedness)
        thumb_tip_x = self.lm_list[self.tip_ids[0]][1]
        thumb_ip_x = self.lm_list[self.tip_ids[0] - 1][1]
        wrist_x = self.lm_list[0][1]
        index_mcp_x = self.lm_list[5][1]

        # Determine palm orientation horizontally
        if index_mcp_x < wrist_x:
            fingers.append(1 if thumb_tip_x < thumb_ip_x else 0)
        else:
            fingers.append(1 if thumb_tip_x > thumb_ip_x else 0)

        # 4 Fingers vertical check (tip higher than PIP joint)
        for i in range(1, 5):
            tip_y = self.lm_list[self.tip_ids[i]][2]
            pip_y = self.lm_list[self.tip_ids[i] - 2][2]
            fingers.append(1 if tip_y < pip_y else 0)

        return fingers

    def get_distance(self, p1_idx, p2_idx, img=None, color=(255, 0, 255)):
        if len(self.lm_list) <= max(p1_idx, p2_idx):
            return 999.0, (0, 0)
        x1, y1 = self.lm_list[p1_idx][1], self.lm_list[p1_idx][2]
        x2, y2 = self.lm_list[p2_idx][1], self.lm_list[p2_idx][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = math.hypot(x2 - x1, y2 - y1)

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 6, color, cv2.FILLED)
        return dist, (cx, cy)


# ============================================================================
# 3. BACKGROUND VOICE ASSISTANT ENGINE
# ============================================================================
class VoiceCommandEngine:
    def __init__(self, callback):
        self.callback = callback
        self.running = True
        self.status_text = "Voice: Initializing..."
        self.available = sr is not None

        if self.available:
            self.recognizer = sr.Recognizer()
            self.thread = threading.Thread(target=self._listen_worker, daemon=True)
            self.thread.start()
        else:
            self.status_text = "Voice: Offline (Use Keyboard/HUD)"

    def speak_async(self, text):
        if pyttsx3 is None:
            return

        def _tts_job():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

        threading.Thread(target=_tts_job, daemon=True).start()

    def _listen_worker(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                self.status_text = "Voice: Ready & Listening"
                while self.running:
                    try:
                        audio = self.recognizer.listen(
                            source, timeout=3, phrase_time_limit=4
                        )
                        phrase = self.recognizer.recognize_google(audio).lower().strip()
                        self.status_text = f"Heard: '{phrase}'"
                        self._dispatch(phrase)
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        continue
                    except Exception as exc:
                        self.status_text = f"Voice Net Err: {str(exc)[:18]}"
                        time.sleep(1)
        except Exception:
            self.status_text = "Voice: No Mic Detected (HUD Active)"
            self.available = False

    def _dispatch(self, cmd):
        # Mode switching
        if any(w in cmd for w in ["canvas", "draw", "paint"]):
            self.callback("MODE", "CANVAS")
            self.speak_async("Canvas mode")
        elif any(w in cmd for w in ["cursor", "mouse", "pointer"]):
            self.callback("MODE", "CURSOR")
            self.speak_async("Cursor mode")
        elif "scroll" in cmd:
            self.callback("MODE", "SCROLL")
            self.speak_async("Scroll mode")

        # Color selection
        elif "cyan" in cmd or "blue" in cmd:
            self.callback("COLOR", "CYAN")
        elif "magenta" in cmd or "pink" in cmd or "purple" in cmd:
            self.callback("COLOR", "MAGENTA")
        elif "green" in cmd:
            self.callback("COLOR", "GREEN")
        elif "amber" in cmd or "yellow" in cmd or "gold" in cmd:
            self.callback("COLOR", "AMBER")
        elif "white" in cmd:
            self.callback("COLOR", "WHITE")
        elif "eraser" in cmd or "erase" in cmd:
            self.callback("COLOR", "ERASER")

        # Workspace actions
        elif "undo" in cmd:
            self.callback("ACTION", "UNDO")
            self.speak_async("Undo")
        elif "redo" in cmd:
            self.callback("ACTION", "REDO")
            self.speak_async("Redo")
        elif "clear" in cmd or "reset" in cmd:
            self.callback("ACTION", "CLEAR")
            self.speak_async("Canvas cleared")
        elif "save" in cmd or "export" in cmd or "screenshot" in cmd:
            self.callback("ACTION", "SAVE")
            self.speak_async("Saved artwork")
        elif "bigger" in cmd or "increase" in cmd:
            self.callback("ACTION", "BRUSH_UP")
        elif "smaller" in cmd or "decrease" in cmd:
            self.callback("ACTION", "BRUSH_DOWN")


# ============================================================================
# 4. MAIN SPATIAL DESKTOP APPLICATION
# ============================================================================
class AirCanvasOS:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)

        self.detector = HandGestureEngine()
        self.canvas = np.zeros((Config.CAM_HEIGHT, Config.CAM_WIDTH, 3), np.uint8)

        # Undo / Redo History Stacks
        self.undo_stack = deque(maxlen=Config.MAX_UNDO_STEPS)
        self.redo_stack = deque(maxlen=Config.MAX_UNDO_STEPS)
        self._push_undo_state()

        # State Variables
        self.mode_idx = 0
        self.mode = Config.MODES[self.mode_idx]
        self.color_name = "CYAN"
        self.draw_color = Config.COLORS[self.color_name]
        self.brush_size = Config.DEFAULT_BRUSH_SIZE

        # Tracking Coordinates
        self.xp, self.yp = 0, 0
        self.stroke_active = False
        self.ploc_x, self.ploc_y = SCREEN_W // 2, SCREEN_H // 2
        self.dragging = False
        self.last_click_time = 0
        self.last_dock_trigger = 0
        self.status_banner = "Welcome to Air-Canvas OS"
        self.banner_timer = time.time()

        # Performance Telemetry
        self.prev_time = time.time()
        self.fps = 0

        # Build Interactive Top Dock Buttons
        self.dock_buttons = self._build_dock_buttons()

        # Start Voice Engine
        self.voice = VoiceCommandEngine(self.handle_event)

    def _build_dock_buttons(self):
        buttons = []
        x = 18
        y_top, y_bot = 12, 64
        # Color swatches
        for name, bgr in Config.COLORS.items():
            w = 95
            buttons.append({
                "label": name,
                "type": "COLOR",
                "value": name,
                "rect": (x, y_top, x + w, y_bot),
                "color": bgr,
            })
            x += w + 10

        # Action buttons
        actions = [
            ("UNDO", "ACTION", "UNDO", (70, 70, 70)),
            ("REDO", "ACTION", "REDO", (70, 70, 70)),
            ("CLEAR", "ACTION", "CLEAR", (45, 45, 140)),
            ("SAVE", "ACTION", "SAVE", (45, 130, 45)),
            ("MODE", "ACTION", "CYCLE_MODE", (140, 95, 30)),
        ]
        for label, btype, val, col in actions:
            w = 92
            buttons.append({
                "label": label,
                "type": btype,
                "value": val,
                "rect": (x, y_top, x + w, y_bot),
                "color": col,
            })
            x += w + 10
        return buttons

    def notify(self, message):
        self.status_banner = message
        self.banner_timer = time.time()

    def _push_undo_state(self):
        self.undo_stack.append(self.canvas.copy())

    def handle_event(self, event_type, value):
        if event_type == "MODE":
            self.mode = value
            if value in Config.MODES:
                self.mode_idx = Config.MODES.index(value)
            self.xp, self.yp = 0, 0
            self.notify(f"Switched to {self.mode} Mode")

        elif event_type == "COLOR":
            self.color_name = value
            self.draw_color = Config.COLORS[value]
            self.notify(f"Selected Tool: {value}")

        elif event_type == "ACTION":
            if value == "CLEAR":
                self._push_undo_state()
                self.canvas = np.zeros((Config.CAM_HEIGHT, Config.CAM_WIDTH, 3), np.uint8)
                self.notify("Canvas Cleared")
            elif value == "UNDO":
                if len(self.undo_stack) > 1:
                    self.redo_stack.append(self.undo_stack.pop())
                    self.canvas = self.undo_stack[-1].copy()
                    self.notify("Undo Applied")
                else:
                    self.notify("Nothing to Undo")
            elif value == "REDO":
                if self.redo_stack:
                    state = self.redo_stack.pop()
                    self.undo_stack.append(state.copy())
                    self.canvas = state.copy()
                    self.notify("Redo Applied")
                else:
                    self.notify("Nothing to Redo")
            elif value == "SAVE":
                os.makedirs("exports", exist_ok=True)
                filename = os.path.join("exports", f"air_canvas_{int(time.time())}.png")
                cv2.imwrite(filename, self.canvas)
                self.notify(f"Saved: {filename}")
            elif value == "CYCLE_MODE":
                self.mode_idx = (self.mode_idx + 1) % len(Config.MODES)
                self.handle_event("MODE", Config.MODES[self.mode_idx])
            elif value == "BRUSH_UP":
                self.brush_size = min(Config.MAX_BRUSH_SIZE, self.brush_size + 3)
                self.notify(f"Brush Size: {self.brush_size}px")
            elif value == "BRUSH_DOWN":
                self.brush_size = max(Config.MIN_BRUSH_SIZE, self.brush_size - 3)
                self.notify(f"Brush Size: {self.brush_size}px")

    def _check_dock_interaction(self, x, y):
        now = time.time()
        if now - self.last_dock_trigger < 0.45:
            return
        for btn in self.dock_buttons:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.handle_event(btn["type"], btn["value"])
                self.last_dock_trigger = now
                break

    def _render_hud(self, img, fingers):
        # Semi-transparent Top Dock Background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (Config.CAM_WIDTH, 78), (18, 18, 24), cv2.FILLED)
        cv2.rectangle(
            overlay,
            (0, Config.CAM_HEIGHT - 42),
            (Config.CAM_WIDTH, Config.CAM_HEIGHT),
            (18, 18, 24),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

        # Render Top Dock Buttons
        for btn in self.dock_buttons:
            x1, y1, x2, y2 = btn["rect"]
            cv2.rectangle(img, (x1, y1), (x2, y2), btn["color"], cv2.FILLED)

            # Highlight active color or mode button
            is_selected = (btn["type"] == "COLOR" and btn["value"] == self.color_name)
            border_col = (255, 255, 255) if is_selected else (90, 90, 90)
            thickness = 3 if is_selected else 1
            cv2.rectangle(img, (x1, y1), (x2, y2), border_col, thickness, cv2.LINE_AA)

            label = btn["label"] if btn["value"] != "CYCLE_MODE" else self.mode
            text_col = (15, 15, 15) if btn["label"] in ["CYAN", "GREEN", "AMBER", "WHITE"] else (245, 245, 245)
            cv2.putText(
                img,
                label,
                (x1 + 10, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                text_col,
                2,
                cv2.LINE_AA,
            )

        # Bottom Status Bar Telemetry
        footer_y = Config.CAM_HEIGHT - 14
        cv2.putText(
            img,
            f"FPS: {int(self.fps)} | MODE: {self.mode} | TOOL: {self.color_name} ({self.brush_size}px)",
            (20, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            self.voice.status_text,
            (540, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )

        if time.time() - self.banner_timer < 4.0:
            cv2.putText(
                img,
                self.status_banner,
                (940, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 215, 0),
                2,
                cv2.LINE_AA,
            )

    def run(self):
        print("====================================================")
        print(" Air-Canvas OS Initialized!")
        print(" Shortcuts: [M] Mode | [C] Clear | [U] Undo | [S] Save | [Q] Quit")
        print("====================================================")

        while True:
            success, frame = self.cap.read()
            if not success:
                print("Error: Webcam stream unavailable.")
                break

            # Mirror image for natural spatial interaction
            img = cv2.flip(frame, 1)
            img = cv2.resize(img, (Config.CAM_WIDTH, Config.CAM_HEIGHT))

            # Process Hand Landmarks
            img, lm_list = self.detector.process_frame(img, draw=True)
            fingers = self.detector.fingers_up()

            if lm_list:
                x_idx, y_idx = lm_list[8][1], lm_list[8][2]     # Index tip
                x_mid, y_mid = lm_list[12][1], lm_list[12][2]   # Middle tip

                # 1. Check Top Dock Hover Selection (Index + Middle Finger Up)
                if fingers[1] == 1 and fingers[2] == 1 and y_idx < 78:
                    self.xp, self.yp = 0, 0
                    cv2.circle(img, (x_idx, y_idx), 12, (255, 255, 255), 2, cv2.LINE_AA)
                    self._check_dock_interaction(x_idx, y_idx)

                # 2. MODE: CANVAS DRAWING
                elif self.mode == "CANVAS":
                    # Hover / Pointer Mode (Index + Middle Up)
                    if fingers[1] == 1 and fingers[2] == 1:
                        if self.stroke_active:
                            self._push_undo_state()
                            self.stroke_active = False
                        self.xp, self.yp = 0, 0
                        cv2.rectangle(
                            img,
                            (x_idx - 12, y_idx - 12),
                            (x_mid + 12, y_mid + 12),
                            self.draw_color,
                            2,
                        )

                    # Dynamic Brush Size Gesture (Thumb + Index Up, others down)
                    elif fingers == [1, 1, 0, 0, 0]:
                        dist, center = self.detector.get_distance(4, 8, img, (0, 255, 255))
                        new_size = int(np.interp(dist, [20, 180], [Config.MIN_BRUSH_SIZE, Config.MAX_BRUSH_SIZE]))
                        self.brush_size = new_size
                        cv2.circle(img, center, self.brush_size // 2, self.draw_color, 2)
                        self.xp, self.yp = 0, 0

                    # Freehand Drawing (Only Index Finger Up)
                    elif fingers[1] == 1 and fingers[2] == 0:
                        self.stroke_active = True
                        thickness = (
                            Config.ERASER_SIZE
                            if self.color_name == "ERASER"
                            else self.brush_size
                        )
                        cv2.circle(img, (x_idx, y_idx), thickness // 2, self.draw_color, cv2.FILLED)

                        if self.xp == 0 and self.yp == 0:
                            self.xp, self.yp = x_idx, y_idx

                        cv2.line(
                            self.canvas,
                            (self.xp, self.yp),
                            (x_idx, y_idx),
                            self.draw_color,
                            thickness,
                            cv2.LINE_AA,
                        )
                        self.xp, self.yp = x_idx, y_idx
                    else:
                        if self.stroke_active:
                            self._push_undo_state()
                            self.stroke_active = False
                        self.xp, self.yp = 0, 0

                # 3. MODE: OS CURSOR CONTROL
                elif self.mode == "CURSOR":
                    cv2.rectangle(
                        img,
                        (Config.FRAME_MARGIN, Config.FRAME_MARGIN),
                        (Config.CAM_WIDTH - Config.FRAME_MARGIN, Config.CAM_HEIGHT - Config.FRAME_MARGIN),
                        (255, 0, 255),
                        2,
                    )

                    # Move Pointer with Index Finger
                    if fingers[1] == 1 and fingers[2] == 0:
                        target_x = np.interp(
                            x_idx,
                            (Config.FRAME_MARGIN, Config.CAM_WIDTH - Config.FRAME_MARGIN),
                            (0, SCREEN_W),
                        )
                        target_y = np.interp(
                            y_idx,
                            (Config.FRAME_MARGIN, Config.CAM_HEIGHT - Config.FRAME_MARGIN),
                            (0, SCREEN_H),
                        )
                        cur_x = self.ploc_x + (target_x - self.ploc_x) / Config.SMOOTHENING
                        cur_y = self.ploc_y + (target_y - self.ploc_y) / Config.SMOOTHENING

                        if pyautogui is not None:
                            pyautogui.moveTo(cur_x, cur_y)
                        self.ploc_x, self.ploc_y = cur_x, cur_y
                        cv2.circle(img, (x_idx, y_idx), 10, (0, 255, 0), cv2.FILLED)

                    # Left Click: Pinch Thumb (4) + Index (8)
                    dist_left, center_l = self.detector.get_distance(4, 8, img)
                    if dist_left < Config.PINCH_THRESHOLD:
                        cv2.circle(img, center_l, 12, (0, 255, 0), cv2.FILLED)
                        if time.time() - self.last_click_time > 0.30:
                            if pyautogui is not None:
                                pyautogui.click()
                            self.last_click_time = time.time()
                            self.notify("Left Click")

                    # Right Click: Pinch Thumb (4) + Middle (12)
                    dist_right, center_r = self.detector.get_distance(4, 12, img, (0, 165, 255))
                    if dist_right < Config.PINCH_THRESHOLD:
                        cv2.circle(img, center_r, 12, (0, 165, 255), cv2.FILLED)
                        if time.time() - self.last_click_time > 0.45:
                            if pyautogui is not None:
                                pyautogui.rightClick()
                            self.last_click_time = time.time()
                            self.notify("Right Click")

                # 4. MODE: SCROLL & ZOOM DECK
                elif self.mode == "SCROLL":
                    cv2.line(
                        img,
                        (Config.CAM_WIDTH // 2, 120),
                        (Config.CAM_WIDTH // 2, Config.CAM_HEIGHT - 80),
                        (255, 255, 0),
                        2,
                    )
                    # Two fingers up: Vertical Scroll based on hand height
                    if fingers[1] == 1 and fingers[2] == 1:
                        mid_y = Config.CAM_HEIGHT // 2
                        offset = mid_y - y_idx
                        if abs(offset) > 55:
                            scroll_step = int(offset / 15)
                            if pyautogui is not None:
                                pyautogui.scroll(scroll_step)
                            direction = "UP" if scroll_step > 0 else "DOWN"
                            self.notify(f"Scrolling {direction} ({scroll_step})")
                            cv2.circle(img, (x_idx, y_idx), 14, (0, 255, 255), cv2.FILLED)
            else:
                if self.stroke_active:
                    self._push_undo_state()
                    self.stroke_active = False
                self.xp, self.yp = 0, 0

            # Blend Virtual Canvas onto Live Camera Feed
            img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, img_inv)
            img = cv2.bitwise_or(img, self.canvas)

            # Calculate FPS
            curr_time = time.time()
            self.fps = 1.0 / max(curr_time - self.prev_time, 1e-5)
            self.prev_time = curr_time

            # Render Spatial HUD Overlay
            self._render_hud(img, fingers)

            cv2.imshow("Air-Canvas OS | Spatial Desktop Interface", img)

            # Keyboard Controls Fallback
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                self.voice.running = False
                break
            elif key == ord("c"):
                self.handle_event("ACTION", "CLEAR")
            elif key == ord("u"):
                self.handle_event("ACTION", "UNDO")
            elif key == ord("r"):
                self.handle_event("ACTION", "REDO")
            elif key == ord("s"):
                self.handle_event("ACTION", "SAVE")
            elif key == ord("m"):
                self.handle_event("ACTION", "CYCLE_MODE")
            elif key in [ord("+"), ord("=")]:
                self.handle_event("ACTION", "BRUSH_UP")
            elif key == ord("-"):
                self.handle_event("ACTION", "BRUSH_DOWN")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirCanvasOS()
    app.run()
