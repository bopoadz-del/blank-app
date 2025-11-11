"""Test stub for FastAPI when the real package is unavailable."""
from .app import (
    APIKeyDependency,
    APIKeyHeader,
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    Security,
)
from . import status

__all__ = [
    "APIKeyDependency",
    "APIKeyHeader",
    "APIRouter",
    "Depends",
    "FastAPI",
    "HTTPException",
    "Request",
    "Response",
    "Security",
    "status",
]
