"""Image pre/post-processing helpers."""
from __future__ import annotations
import io
import uuid
import math
from typing import Any

import cv2
import numpy as np
from PIL import Image


def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    """Convert raw bytes → OpenCV BGR image."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Unsupported format or corrupt file.")
    return img


def crop_mole(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Return a cropped ROI from the image."""
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def draw_detections(image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes + labels onto a copy of the image."""
    annotated = image.copy()
    RISK_COLORS = {
        "Low": (0, 200, 0),        # Green
        "Moderate": (0, 165, 255),  # Orange
        "High": (0, 0, 220),        # Red
    }
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        bb = det["bounding_box"]
        x1, y1, x2, y2 = int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])
        color = RISK_COLORS.get(det["risk_level"], (180, 180, 180))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['mole_id']} | {det['risk_level']} ({det['confidence']:.0%})"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3), font, 0.5, (255, 255, 255), 1)

    return annotated


def ndarray_to_bytes(image: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode an ndarray image to bytes."""
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return buf.tobytes()


def generate_unique_filename(original_name: str) -> str:
    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else "jpg"
    return f"{uuid.uuid4().hex}.{ext}"
