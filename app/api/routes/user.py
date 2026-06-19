"""
User API routes.
  GET /user/{user_id}/history — paginated scan history
"""
from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.scan import UserHistoryResponse
from app.services.scan_service import get_user_history

logger = logging.getLogger("yolocheck.routes.user")
router = APIRouter(prefix="/user", tags=["User"])


@router.get(
    "/{user_id}/history",
    response_model=UserHistoryResponse,
    summary="Get all scans for a user",
)
async def user_history(user_id: str):
    try:
        result = get_user_history(user_id)
    except Exception:
        logger.exception("Failed to fetch history for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve scan history.",
        )
    return result
