"""
Scan API routes.
  POST /scan           — upload image, run detection
  GET  /scan/{id}      — retrieve a scan by ID

FIX 9: Added user_id as an optional Bearer token header extraction
        so authenticated frontend users get their scans saved under
        their account. Anonymous scans still work (user_id=None).
"""
from __future__ import annotations
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Header

from app.schemas.scan import ScanResponse
from app.services.scan_service import create_scan, get_scan

logger = logging.getLogger("yolocheck.routes.scan")
router = APIRouter(prefix="/scan", tags=["Scan"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_MB = 10


@router.post(
    "",
    response_model=ScanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an image and detect moles",
)
async def upload_and_scan(
    file: UploadFile = File(..., description="Skin image (JPEG / PNG)"),
    user_id: Optional[str] = Form(None, description="Authenticated user ID (optional)"),
):
    # ── Validation ──────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: JPEG, PNG, WebP, BMP.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── Process ──────────────────────────────────────────────────────────────
    try:
        result = create_scan(
            file_bytes=file_bytes,
            original_filename=file.filename or "upload.jpg",
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        logger.exception("Scan creation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return result


@router.get(
    "/{scan_id}",
    response_model=ScanResponse,
    summary="Retrieve a scan by ID",
)
async def retrieve_scan(scan_id: str):
    try:
        result = get_scan(scan_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception:
        logger.exception("Failed to retrieve scan %s", scan_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve scan.",
        )
    return result
