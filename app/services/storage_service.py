"""
Supabase Storage service.
Uploads scan images and returns a public URL.
"""
from __future__ import annotations
import logging
from typing import Optional

from app.db.supabase_client import get_supabase
from app.core.config import get_settings

logger = logging.getLogger("yolocheck.storage")


def upload_scan_image(
    image_bytes: bytes,
    filename: str,
    content_type: str = "image/jpeg",
) -> str:
    settings = get_settings()
    supabase = get_supabase()
    bucket = settings.supabase_storage_bucket
    path = f"scans/{filename}"

    try:
        supabase.storage.from_(bucket).upload(
            path=path,
            file=image_bytes,
            file_options={"content-type": content_type, "upsert": "true"},
        )

        url_response = supabase.storage.from_(bucket).get_public_url(path)

        if isinstance(url_response, str):
            public_url = url_response
        elif isinstance(url_response, dict):
            public_url = (
                url_response.get("publicUrl")
                or url_response.get("publicURL")
                or url_response.get("data", {}).get("publicUrl", "")
            )
        else:
            public_url = str(url_response)

        logger.info("Uploaded image → %s", public_url)
        return public_url

    except Exception as exc:
        logger.error("Storage upload failed: %s", exc)
        raise RuntimeError(f"Image upload failed: {exc}") from exc


def delete_scan_image(filename: str) -> None:
    settings = get_settings()
    supabase = get_supabase()
    path = f"scans/{filename}"
    try:
        supabase.storage.from_(settings.supabase_storage_bucket).remove([path])
        logger.info("Deleted image: %s", path)
    except Exception as exc:
        logger.warning("Could not delete image '%s': %s", path, exc)