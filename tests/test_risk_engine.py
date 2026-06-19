"""Unit tests for the ABCD risk engine."""
import numpy as np
import pytest
from app.services.risk_engine import compute_abcd, _classify_risk


def make_roi(h: int = 80, w: int = 80, color: tuple = (100, 80, 60)) -> np.ndarray:
    """Create a solid-colour BGR test ROI."""
    roi = np.full((h, w, 3), color, dtype=np.uint8)
    return roi


class TestClassifyRisk:
    def test_low(self):
        assert _classify_risk(0) == "Low"
        assert _classify_risk(4) == "Low"

    def test_moderate(self):
        assert _classify_risk(5) == "Moderate"
        assert _classify_risk(8) == "Moderate"

    def test_high(self):
        assert _classify_risk(9) == "High"
        assert _classify_risk(12) == "High"


class TestComputeABCD:
    def test_returns_result(self):
        roi = make_roi()
        result = compute_abcd(roi, 80, 80, 640, 480)
        assert result.risk_level in ("Low", "Moderate", "High")
        assert 0 <= result.total_score <= 12

    def test_tiny_mole_low_diameter(self):
        roi = make_roi(10, 10)
        result = compute_abcd(roi, 10, 10, 1920, 1080)
        # Very small mole — diameter score should be near 0
        assert result.diameter < 1.5

    def test_large_mole_high_diameter(self):
        roi = make_roi(400, 400)
        result = compute_abcd(roi, 400, 400, 640, 480)
        assert result.diameter > 1.0

    def test_scores_in_range(self):
        roi = make_roi()
        result = compute_abcd(roi, 80, 80, 640, 480)
        for score in (result.asymmetry, result.border, result.color, result.diameter):
            assert 0 <= score <= 3

    def test_notes_are_strings(self):
        roi = make_roi()
        result = compute_abcd(roi, 80, 80, 640, 480)
        assert isinstance(result.asymmetry_note, str)
        assert isinstance(result.border_note, str)
        assert isinstance(result.color_note, str)
        assert isinstance(result.diameter_note, str)
