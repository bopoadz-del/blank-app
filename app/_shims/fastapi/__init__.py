"""Lightweight fallback implementation for the FastAPI package."""

from .app import FastAPI, APIRouter, Depends, Security, HTTPException, Request
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
