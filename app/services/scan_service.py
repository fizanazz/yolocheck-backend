"""
Scan service — orchestrates detection, Supabase Storage upload, and DB persistence.
"""
from __future__ import annotations
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.services.yolo_service import detect_moles
from app.services.storage_service import upload_scan_image
from app.db.supabase_client import get_supabase
from app.utils.image_utils import read_image_bytes, generate_unique_filename

logger = logging.getLogger("yolocheck.scan")


def _highest_risk(detections: list[dict]) -> str:
    priority = {"High": 3, "Moderate": 2, "Low": 1}
    if not detections:
        return "Low"
    return max(detections, key=lambda d: priority.get(d["risk_level"], 0))["risk_level"]


def create_scan(
    file_bytes: bytes,
    original_filename: str,
    user_id: Optional[str] = None,
) -> dict[str, Any]:

    # 1. Decode image
    image = read_image_bytes(file_bytes)

    # 2. Run YOLO detection
    detections = detect_moles(image)
    highest    = _highest_risk(detections)

    # 3. Upload image to Supabase Storage
    unique_name  = generate_unique_filename(original_filename)
    content_type = "image/jpeg"
    if original_filename.lower().endswith(".png"):
        content_type = "image/png"
    elif original_filename.lower().endswith(".webp"):
        content_type = "image/webp"

    try:
        image_url = upload_scan_image(file_bytes, unique_name, content_type)
        logger.info("Image uploaded → %s", image_url)
    except Exception as exc:
        logger.error("Image upload failed: %s", exc)
        image_url = ""

    # 4. Persist scan to Supabase DB
    scan_id = str(uuid.uuid4())
    now     = datetime.now(timezone.utc).isoformat()

    try:
        supabase = get_supabase()

        scan_row = {
            "id":                   scan_id,
            "user_id":              user_id,
            "image_url":            image_url,
            "image_filename":       unique_name,
            "total_moles_detected": len(detections),
            "highest_risk":         highest,
            "created_at":           now,
        }
        supabase.table("scans").insert(scan_row).execute()
        logger.info("Scan %s saved to DB.", scan_id)

        # Save individual detection rows
        det_rows = []
        for det in detections:
            bb   = det["bounding_box"]
            abcd = det["abcd"]
            det_rows.append({
                "id":              str(uuid.uuid4()),
                "scan_id":         scan_id,
                "mole_id":         det["mole_id"],
                "confidence":      det["confidence"],
                "label":           det.get("label", "unknown"),  # ← FIXED
                "bbox_x1":         bb["x1"],
                "bbox_y1":         bb["y1"],
                "bbox_x2":         bb["x2"],
                "bbox_y2":         bb["y2"],
                "bbox_width":      bb["width"],
                "bbox_height":     bb["height"],
                "abcd_asymmetry":  abcd["asymmetry"],
                "abcd_border":     abcd["border"],
                "abcd_color":      abcd["color"],
                "abcd_diameter":   abcd["diameter"],
                "abcd_total":      abcd["total_score"],
                "asymmetry_note":  abcd["asymmetry_note"],
                "border_note":     abcd["border_note"],
                "color_note":      abcd["color_note"],
                "diameter_note":   abcd["diameter_note"],
                "risk_level":      det["risk_level"],
                "risk_score":      det["risk_score"],
            })
        if det_rows:
            supabase.table("detections").insert(det_rows).execute()
            logger.info("Saved %d detection(s) to DB.", len(det_rows))

    except Exception as exc:
        logger.error("DB save failed: %s", exc)

    return {
        "scan_id":               scan_id,
        "user_id":               user_id,
        "image_url":             image_url,
        "total_moles_detected":  len(detections),
        "detections":            detections,
        "highest_risk":          highest,
        "created_at":            now,
    }


def get_scan(scan_id: str) -> dict[str, Any]:
    try:
        supabase  = get_supabase()
        scan_resp = supabase.table("scans").select("*").eq("id", scan_id).single().execute()

        if not scan_resp.data:
            raise ValueError(f"Scan '{scan_id}' not found.")

        det_resp = (
            supabase.table("detections")
            .select("*")
            .eq("scan_id", scan_id)
            .order("mole_id")
            .execute()
        )

        detections = []
        for r in (det_resp.data or []):
            detections.append({
                "mole_id": r["mole_id"],
                "bounding_box": {
                    "x1":     r["bbox_x1"], "y1": r["bbox_y1"],
                    "x2":     r["bbox_x2"], "y2": r["bbox_y2"],
                    "width":  r["bbox_width"], "height": r["bbox_height"],
                },
                "confidence": r["confidence"],
                "label":      r.get("label", "unknown"),  # reads real label from DB
                "abcd": {
                    "asymmetry":      r["abcd_asymmetry"],
                    "border":         r["abcd_border"],
                    "color":          r["abcd_color"],
                    "diameter":       r["abcd_diameter"],
                    "total_score":    r["abcd_total"],
                    "asymmetry_note": r.get("asymmetry_note", ""),
                    "border_note":    r.get("border_note", ""),
                    "color_note":     r.get("color_note", ""),
                    "diameter_note":  r.get("diameter_note", ""),
                },
                "risk_level": r["risk_level"],
                "risk_score": r["risk_score"],
            })

        scan = scan_resp.data
        return {
            "scan_id":               scan["id"],
            "user_id":               scan.get("user_id"),
            "image_url":             scan["image_url"],
            "total_moles_detected":  scan["total_moles_detected"],
            "detections":            detections,
            "highest_risk":          scan["highest_risk"],
            "created_at":            scan["created_at"],
        }

    except ValueError:
        raise
    except Exception as exc:
        logger.error("get_scan failed: %s", exc)
        raise ValueError(f"Could not retrieve scan '{scan_id}'.")


def get_user_history(user_id: str) -> dict[str, Any]:
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("scans")
            .select("id, image_url, total_moles_detected, highest_risk, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        scans = [
            {
                "scan_id":               s["id"],
                "image_url":             s["image_url"],
                "total_moles_detected":  s["total_moles_detected"],
                "highest_risk":          s["highest_risk"],
                "created_at":            s["created_at"],
            }
            for s in (resp.data or [])
        ]
        return {
            "user_id":     user_id,
            "total_scans": len(scans),
            "scans":       scans,
        }

    except Exception as exc:
        logger.error("get_user_history failed: %s", exc)
        return {"user_id": user_id, "total_scans": 0, "scans": []}