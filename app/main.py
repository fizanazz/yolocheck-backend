"""
YOLOCheck — FastAPI application entrypoint.
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.exceptions import (
    YoloCheckException,
    yolocheck_exception_handler,
    generic_exception_handler,
)
from app.services.yolo_service import load_model, detect_moles
from app.api.routes import health, scan, user, chat

logger = logging.getLogger("yolocheck.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    settings = get_settings()
    setup_logging(debug=settings.debug)

    # Load model
    load_model()

    # ── Warmup — run dummy prediction so first real request is fast ──────────
    try:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        detect_moles(dummy)
        logger.info("Model warmup complete — first request will be fast.")
    except Exception as e:
        logger.warning("Warmup failed (non-critical): %s", e)

    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "YOLOCheck — Fast and Accurate Mole Detection System. "
            "Detects moles via YOLOv11, performs ABCD skin analysis, "
            "and provides an AI-powered educational health assistant."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────
    app.add_exception_handler(YoloCheckException, yolocheck_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(scan.router)
    app.include_router(user.router)
    app.include_router(chat.router)

    # ── Serve uploaded images locally ─────────────────────────────────────────
    uploads_dir = "./uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

    return app


app = create_app()