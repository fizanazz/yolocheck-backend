"""
YOLOv11 inference service.
Loads the model once at startup and exposes a `detect_moles` function.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from app.core.config import get_settings
from app.services.risk_engine import compute_abcd
from app.utils.image_utils import crop_mole

logger = logging.getLogger("yolocheck.yolo")

# Module-level singleton — loaded once per process
_model: YOLO | None = None


def load_model() -> None:
    """Load YOLOv11 model into memory (call once at startup)."""
    global _model
    settings = get_settings()
    model_path = Path(settings.model_path)

    if not model_path.exists():
        logger.warning(
            "Model file '%s' not found. YOLO inference will use stub mode.", model_path
        )
        _model = None
        return

    logger.info("Loading YOLOv11 model from '%s' …", model_path)
    _model = YOLO(str(model_path))
    logger.info("Model loaded successfully.")


def _stub_detections(image: np.ndarray) -> list[dict[str, Any]]:
    """
    Return a deterministic fake detection for demo / testing purposes
    when no real model file is available.
    """
    h, w = image.shape[:2]
    return [
        {
            "x1": w * 0.25, "y1": h * 0.25,
            "x2": w * 0.45, "y2": h * 0.50,
            "confidence": 0.82,
            "label": "benign",
        },
        {
            "x1": w * 0.60, "y1": h * 0.30,
            "x2": w * 0.75, "y2": h * 0.55,
            "confidence": 0.67,
            "label": "benign",
        },
    ]


def detect_moles(image: np.ndarray) -> list[dict[str, Any]]:
    """
    Run YOLOv11 inference on a BGR numpy image.

    Returns a list of mole detection dicts, each containing:
      - mole_id, bounding_box, confidence, label, abcd, risk_level, risk_score
    """
    settings = get_settings()
    image_h, image_w = image.shape[:2]

    # ── Run inference ──────────────────────────────────────────────────────────
    if _model is not None:
        results = _model.predict(
            source=image,
            conf=settings.yolo_confidence_threshold,
            iou=settings.yolo_iou_threshold,
            verbose=False,
        )
        raw_boxes: list[dict[str, Any]] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy  = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = results[0].names[cls]  # "benign" or "malignant"
                raw_boxes.append({
                    "x1": xyxy[0], "y1": xyxy[1],
                    "x2": xyxy[2], "y2": xyxy[3],
                    "confidence": conf,
                    "label": label,
                })
    else:
        logger.debug("Model not loaded — using stub detections.")
        raw_boxes = _stub_detections(image)

    # Sort highest confidence first
    raw_boxes.sort(key=lambda b: b["confidence"], reverse=True)

    # ── ABCD analysis per detection ────────────────────────────────────────────
    detections: list[dict[str, Any]] = []
    for idx, box in enumerate(raw_boxes, start=1):
        x1, y1, x2, y2 = (
            int(box["x1"]), int(box["y1"]),
            int(box["x2"]), int(box["y2"]),
        )
        bw = x2 - x1
        bh = y2 - y1

        roi  = crop_mole(image, x1, y1, x2, y2)
        abcd = compute_abcd(roi, bw, bh, image_w, image_h)

        detections.append({
            "mole_id": f"Mole #{idx}",
            "bounding_box": {
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "width": float(bw),
                "height": float(bh),
            },
            "confidence": round(box["confidence"], 4),
            "label":      box.get("label", "unknown"),  # "benign" or "malignant"
            "abcd": {
                "asymmetry":      abcd.asymmetry,
                "border":         abcd.border,
                "color":          abcd.color,
                "diameter":       abcd.diameter,
                "total_score":    abcd.total_score,
                "asymmetry_note": abcd.asymmetry_note,
                "border_note":    abcd.border_note,
                "color_note":     abcd.color_note,
                "diameter_note":  abcd.diameter_note,
            },
            "risk_level": abcd.risk_level,
            "risk_score": abcd.total_score,
        })

    return detections