"""Lightweight pydantic-settings compatibility layer for offline testing."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BaseSettings(BaseModel):
    """Minimal drop-in replacement implementing the few hooks the app needs."""

    class Config:
        env_file = None
        case_sensitive = False

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)


__all__ = ["BaseSettings"]
