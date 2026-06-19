"""
Pydantic schemas for scan-related request / response bodies.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── ABCD Analysis ─────────────────────────────────────────────────────────────

class ABCDAnalysis(BaseModel):
    asymmetry:      float = Field(..., ge=0, le=3, description="Asymmetry score 0–3")
    border:         float = Field(..., ge=0, le=3, description="Border score 0–3")
    color:          float = Field(..., ge=0, le=3, description="Color score 0–3")
    diameter:       float = Field(..., ge=0, le=3, description="Diameter score 0–3")
    total_score:    float = Field(..., ge=0, le=12)
    asymmetry_note: str
    border_note:    str
    color_note:     str
    diameter_note:  str


# ── Bounding Box ──────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1:     float
    y1:     float
    x2:     float
    y2:     float
    width:  float
    height: float


# ── Mole Detection ────────────────────────────────────────────────────────────

class MoleDetection(BaseModel):
    mole_id:      str   = Field(..., example="Mole #1")
    bounding_box: BoundingBox
    confidence:   float = Field(..., ge=0, le=1)
    label:        str   = Field("unknown", example="benign")  # ← NEW
    abcd:         ABCDAnalysis
    risk_level:   str   = Field(..., example="Low")
    risk_score:   float


# ── Scan Response ─────────────────────────────────────────────────────────────

class ScanResponse(BaseModel):
    scan_id:               str
    user_id:               Optional[str]
    image_url:             str
    total_moles_detected:  int
    detections:            list[MoleDetection]
    highest_risk:          str
    created_at:            datetime


class ScanSummary(BaseModel):
    scan_id:               str
    image_url:             str
    total_moles_detected:  int
    highest_risk:          str
    created_at:            datetime


class UserHistoryResponse(BaseModel):
    user_id:     str
    total_scans: int
    scans:       list[ScanSummary]