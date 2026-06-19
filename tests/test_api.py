"""
Integration-style tests for the FastAPI endpoints.
Uses TestClient (no real Supabase / Gemini needed — services are mocked).
"""
import io
import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Patch external dependencies before importing the app
MOCK_SCAN_RESULT = {
    "scan_id": "test-scan-123",
    "user_id": None,
    "image_url": "https://example.com/scan.jpg",
    "total_moles_detected": 1,
    "detections": [
        {
            "mole_id": "Mole #1",
            "bounding_box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "width": 40, "height": 40},
            "confidence": 0.85,
            "abcd": {
                "asymmetry": 0.5, "border": 0.8, "color": 0.3, "diameter": 0.2,
                "total_score": 1.8,
                "asymmetry_note": "Roughly symmetric.",
                "border_note": "Smooth border.",
                "color_note": "Uniform colour.",
                "diameter_note": "Small lesion.",
            },
            "risk_level": "Low",
            "risk_score": 1.8,
        }
    ],
    "highest_risk": "Low",
    "created_at": "2024-01-01T00:00:00+00:00",
}


@pytest.fixture
def client():
    with (
        patch("app.services.yolo_service.load_model"),
        patch("app.core.config.Settings.model_validate", return_value=MagicMock()),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c


def _make_jpeg_bytes() -> bytes:
    """Create a minimal valid JPEG in memory."""
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestScanEndpoint:
    @patch("app.api.routes.scan.create_scan", return_value=MOCK_SCAN_RESULT)
    def test_upload_scan_success(self, mock_create, client):
        image_bytes = _make_jpeg_bytes()
        resp = client.post(
            "/scan",
            files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["scan_id"] == "test-scan-123"
        assert data["total_moles_detected"] == 1

    def test_upload_wrong_type(self, client):
        resp = client.post(
            "/scan",
            files={"file": ("test.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert resp.status_code == 415

    def test_upload_empty_file(self, client):
        resp = client.post(
            "/scan",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert resp.status_code == 400

    @patch("app.api.routes.scan.get_scan", return_value=MOCK_SCAN_RESULT)
    def test_get_scan(self, mock_get, client):
        resp = client.get("/scan/test-scan-123")
        assert resp.status_code == 200
        assert resp.json()["scan_id"] == "test-scan-123"

    @patch("app.api.routes.scan.get_scan", side_effect=ValueError("Not found"))
    def test_get_scan_not_found(self, mock_get, client):
        resp = client.get("/scan/does-not-exist")
        assert resp.status_code == 404


class TestChatEndpoint:
    @patch("app.api.routes.chat.chat", return_value="Here is some educational info. ⚠️ not a medical diagnosis.")
    def test_chat_success(self, mock_chat, client):
        resp = client.post("/chat", json={"message": "What is the ABCD rule?"})
        assert resp.status_code == 200
        assert "reply" in resp.json()
        assert "disclaimer" in resp.json()

    def test_chat_empty_message(self, client):
        resp = client.post("/chat", json={"message": "   "})
        assert resp.status_code == 400
