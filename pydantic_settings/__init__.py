"""Minimal subset of pydantic-settings for tests."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BaseSettings(BaseModel):
    class Config:
        env_file = None
        case_sensitive = False

    def __init__(self, **values: Any):
        super().__init__(**values)
