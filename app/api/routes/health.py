"""Health-check endpoint."""
from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter(tags=["Health"])

@router.get("/health", summary="Health check")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "supabase_url": settings.supabase_url,
        "anon_key_prefix": settings.supabase_anon_key[:20],
        "service_key_prefix": settings.supabase_service_role_key[:20],
    }