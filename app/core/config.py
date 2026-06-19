"""
Application configuration — loaded from environment variables / .env file.

FIX 6: allowed_origins was typed as list[str] which works fine when set
        in code, but when loaded from a .env file as a comma-separated
        string (e.g. "http://localhost:3000,http://localhost:5173") Pydantic
        cannot automatically parse it into a list and raises a validation error.
        Added a validator to handle both formats.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ────────────────────────────────────────────────────────────────────
    app_name: str = "YOLOCheck"
    app_version: str = "1.0.0"
    debug: bool = False
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── Supabase ───────────────────────────────────────────────────────────────
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "scan-images"

    # ── Google Gemini ──────────────────────────────────────────────────────────
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"

    # ── YOLO ───────────────────────────────────────────────────────────────────
    model_path: Path = Path("ml/best.pt")
    yolo_confidence_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45

    # FIX 6: parse comma-separated string from .env into a list
    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
