"""Lightweight FastAPI compatibility layer for offline testing."""
from .app import APIRouter, Depends, FastAPI, HTTPException, Request, Security
from .status import status

__all__ = [
    "FastAPI",
    "APIRouter",
    "Depends",
    "Security",
    "HTTPException",
    "Request",
    "status",
]
