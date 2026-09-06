"""Per-camera ROI (pitch polygon) masking.

Polygons live in config/cameras.json as normalized [0-1] coords:
    {"<camera_tag>": {"roi": [[x, y], ...]}, ...}
A camera without a polygon passes through unmasked.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "cameras.json"


def load_rois() -> dict:
    if CONFIG.exists():
        return {k: v.get("roi") for k, v in json.loads(CONFIG.read_text()).items() if v.get("roi")}
    return {}


def apply_roi(frame_bgr: np.ndarray, roi_norm: list | None) -> np.ndarray:
    """Black out everything outside the polygon. No-op when roi_norm is falsy."""
    if not roi_norm:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in roi_norm], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
