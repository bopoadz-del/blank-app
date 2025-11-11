"""Lightweight FastAPI compatibility layer used in environments without FastAPI."""
from __future__ import annotations

from .app import (  # noqa: F401
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
)
from .status import status  # noqa: F401

__all__ = [
    "FastAPI",
    "APIRouter",
    "Depends",
    "Security",
    "HTTPException",
    "Request",
    "status",
]
