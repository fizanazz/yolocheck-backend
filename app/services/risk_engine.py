"""
ABCD Risk Engine — rule-based ABCD dermoscopy scoring.

ABCD Rule of Dermoscopy (simplified):
  A – Asymmetry   (0–3)
  B – Border      (0–3)
  C – Color       (0–3)
  D – Diameter    (0–3)
  Total Score     (0–12)

Risk Classification:
  Low      :  0 – 4
  Moderate :  5 – 8
  High     :  9 – 12
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class ABCDResult:
    asymmetry: float
    border: float
    color: float
    diameter: float
    total_score: float
    risk_level: str
    asymmetry_note: str
    border_note: str
    color_note: str
    diameter_note: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _score_asymmetry(roi: np.ndarray) -> tuple[float, str]:
    """
    Flip the ROI across both axes and compare pixel-level dissimilarity.
    Returns a score in [0, 3].
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    h, w = gray.shape
    if h < 4 or w < 4:
        return 1.5, "Insufficient region size to assess asymmetry."

    # Horizontal flip comparison
    flipped_h = cv2.flip(gray, 1)
    diff_h = np.mean(np.abs(gray.astype(float) - flipped_h.astype(float))) / 255.0

    # Vertical flip comparison
    flipped_v = cv2.flip(gray, 0)
    diff_v = np.mean(np.abs(gray.astype(float) - flipped_v.astype(float))) / 255.0

    raw = (diff_h + diff_v) / 2.0  # 0..1
    score = _clamp(raw * 6, 0, 3)  # scale to 0–3

    if score < 1:
        note = "The mole appears roughly symmetric — a reassuring sign."
    elif score < 2:
        note = "Mild asymmetry detected; one half does not mirror the other."
    else:
        note = "Significant asymmetry present; this warrants professional evaluation."

    return round(score, 2), note


def _score_border(roi: np.ndarray) -> tuple[float, str]:
    """
    Analyse border irregularity using edge detection.
    A jagged perimeter gets a higher score.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    if gray.shape[0] < 4 or gray.shape[1] < 4:
        return 1.5, "Insufficient region size to assess border."

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Edge pixel density as a fraction of total perimeter pixels
    h, w = gray.shape
    perimeter_pixels = 2 * (h + w)
    edge_count = np.sum(edges > 0)
    density = edge_count / max(perimeter_pixels, 1)

    score = _clamp(density * 15, 0, 3)

    if score < 1:
        note = "Well-defined, smooth border — generally a benign indicator."
    elif score < 2:
        note = "Slightly irregular or notched borders observed."
    else:
        note = "Highly irregular or poorly defined borders — professional review recommended."

    return round(score, 2), note


def _score_color(roi: np.ndarray) -> tuple[float, str]:
    """
    Estimate colour variegation in the HSV colour space.
    Multiple distinct hues raise the score.
    """
    if len(roi.shape) < 3 or roi.shape[0] < 4 or roi.shape[1] < 4:
        return 1.5, "Insufficient region size to assess colour."

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_std = float(np.std(hsv[:, :, 0]))    # hue std dev (0–180)
    sat_std = float(np.std(hsv[:, :, 1]))    # saturation std dev (0–255)

    # Normalise: hue_std up to ~60 maps to max score
    score = _clamp((hue_std / 60.0) * 2 + (sat_std / 255.0), 0, 3)

    if score < 1:
        note = "Uniform colouring — consistent pigmentation detected."
    elif score < 2:
        note = "Mild colour variation present within the lesion."
    else:
        note = "Multiple colours or uneven pigmentation detected — further assessment advised."

    return round(score, 2), note


def _score_diameter(bbox_width_px: float, bbox_height_px: float,
                    image_width: int, image_height: int) -> tuple[float, str]:
    """
    Estimate lesion diameter relative to the image frame.
    Clinical threshold is ~6 mm; we approximate using relative size.
    A mole covering >2% of image area is flagged as potentially large.
    """
    image_area = image_width * image_height
    mole_area = bbox_width_px * bbox_height_px
    ratio = mole_area / max(image_area, 1)

    # Map ratio to 0–3: ratio of 0.005 → ~1, 0.02 → ~3
    score = _clamp(ratio / 0.02 * 3, 0, 3)

    if score < 1:
        note = "Small lesion — diameter appears within normal range."
    elif score < 2:
        note = "Medium-sized lesion; diameter approaching the 6 mm clinical threshold."
    else:
        note = "Large lesion detected; diameter exceeds typical benign size — evaluation advised."

    return round(score, 2), note


# ── Public API ────────────────────────────────────────────────────────────────

def compute_abcd(
    roi: np.ndarray,
    bbox_w: float,
    bbox_h: float,
    image_w: int,
    image_h: int,
) -> ABCDResult:
    """
    Run all four ABCD criteria and return a structured result.

    Args:
        roi:       Cropped numpy array of the mole region (BGR).
        bbox_w:    Bounding-box pixel width.
        bbox_h:    Bounding-box pixel height.
        image_w:   Full image pixel width.
        image_h:   Full image pixel height.
    """
    a_score, a_note = _score_asymmetry(roi)
    b_score, b_note = _score_border(roi)
    c_score, c_note = _score_color(roi)
    d_score, d_note = _score_diameter(bbox_w, bbox_h, image_w, image_h)

    total = round(a_score + b_score + c_score + d_score, 2)
    risk = _classify_risk(total)

    return ABCDResult(
        asymmetry=a_score,
        border=b_score,
        color=c_score,
        diameter=d_score,
        total_score=total,
        risk_level=risk,
        asymmetry_note=a_note,
        border_note=b_note,
        color_note=c_note,
        diameter_note=d_note,
    )


def _classify_risk(total: float) -> str:
    if total <= 4:
        return "Low"
    elif total <= 8:
        return "Moderate"
    else:
        return "High"
