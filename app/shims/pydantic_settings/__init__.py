"""Minimal `pydantic-settings` compatible API for environments without the dependency."""
from __future__ import annotations

from pydantic import BaseModel


class BaseSettings(BaseModel):
    """Simplified stand-in for Pydantic BaseSettings."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def model_dump(self, *args, **kwargs):  # pragma: no cover - delegate to BaseModel
        return super().model_dump(*args, **kwargs)


__all__ = ["BaseSettings"]
