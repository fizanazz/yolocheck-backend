"""
app/core/database.py
Supabase client — single shared instance injected via FastAPI dependency.
Uses the service-role key for backend operations (bypasses RLS).
"""

from functools import lru_cache
from supabase import create_client, Client
from app.core.config import get_settings


@lru_cache
def get_supabase() -> Client:
    """Return a cached Supabase client using the service-role key."""
    cfg = get_settings()
    return create_client(cfg.supabase_url, cfg.supabase_service_key)
