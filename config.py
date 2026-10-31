"""Configuration constants for Air-Canvas OS."""
import pyautogui

try:
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01
    SCREEN_W, SCREEN_H = pyautogui.size()
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080


class Config:
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    FPS_TARGET = 30

    SMOOTHENING = 5.5
    FRAME_MARGIN = 110
    PINCH_THRESHOLD = 36

    DEFAULT_BRUSH_SIZE = 8
    MIN_BRUSH_SIZE = 2
    MAX_BRUSH_SIZE = 45
    ERASER_SIZE = 55
    MAX_UNDO_STEPS = 15

    COLORS = {
        "CYAN": (255, 255, 0),
        "MAGENTA": (255, 0, 255),
        "GREEN": (0, 255, 128),
        "AMBER": (0, 191, 255),
        "WHITE": (255, 255, 255),
        "ERASER": (0, 0, 0),
    }

    MODES = ["CANVAS", "CURSOR", "SCROLL"]
