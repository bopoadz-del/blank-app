"""Minimal `pydantic-settings` compatible API for environments without the dependency."""
from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Dict

if find_spec("pydantic") is not None:  # pragma: no cover - exercised when dependency is present
    from pydantic import BaseModel
else:  # pragma: no cover - used in shim-only environments
    class BaseModel:  # type: ignore[too-few-public-methods]
        """Simple substitute supporting `model_dump`."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return self.__dict__.copy()


class BaseSettings(BaseModel):
    """Simplified stand-in for Pydantic BaseSettings."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def model_dump(self, *args, **kwargs):  # pragma: no cover - delegate to BaseModel
        return super().model_dump(*args, **kwargs)


__all__ = ["BaseSettings"]
