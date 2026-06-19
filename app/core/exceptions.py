"""
app/core/exceptions.py
Domain-specific exceptions and FastAPI exception handlers.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


class YoloCheckException(Exception):
    """Base exception for all domain errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ScanNotFoundException(YoloCheckException):
    def __init__(self, scan_id: str):
        super().__init__(f"Scan '{scan_id}' not found.", status_code=404)


class UserNotFoundException(YoloCheckException):
    def __init__(self, user_id: str):
        super().__init__(f"User '{user_id}' not found.", status_code=404)


class InvalidImageException(YoloCheckException):
    def __init__(self, reason: str = "Invalid or corrupt image file."):
        super().__init__(reason, status_code=422)


class ImageTooLargeException(YoloCheckException):
    def __init__(self, max_mb: int):
        super().__init__(
            f"Uploaded image exceeds the {max_mb} MB size limit.",
            status_code=413,
        )


class ModelInferenceException(YoloCheckException):
    def __init__(self, detail: str = "YOLO inference failed."):
        super().__init__(detail, status_code=500)


class StorageException(YoloCheckException):
    def __init__(self, detail: str = "Failed to upload image to storage."):
        super().__init__(detail, status_code=500)


# ── FastAPI exception handlers ─────────────────────────────────────────────


async def yolocheck_exception_handler(
    request: Request, exc: YoloCheckException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "status_code": exc.status_code},
    )


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected internal error occurred.",
            "status_code": 500,
        },
    )
