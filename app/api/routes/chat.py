"""
AI Health Assistant chat route.
  POST /chat

FIX 3: Was importing ChatRequest and ChatResponse from app.schemas.scan
        (the wrong module). Now imports from app.schemas.chat which has
        the correct schema with conversation history support.
"""
from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import ChatRequest, ChatResponse   # <-- FIXED import
from app.services.chat_service import chat

logger = logging.getLogger("yolocheck.routes.chat")
router = APIRouter(prefix="/chat", tags=["AI Assistant"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask the AI Health Assistant a question",
)
async def ask_assistant(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    try:
        reply = chat(
            message=request.message,
            scan_id=request.scan_id,
            user_id=request.user_id,
            history=request.history,         # pass conversation history
        )
    except Exception:
        logger.exception("Chat service error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI assistant is temporarily unavailable.",
        )

    return ChatResponse(reply=reply)
