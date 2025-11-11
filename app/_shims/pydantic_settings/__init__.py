"""Lightweight fallback for :mod:`pydantic_settings`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel as _BaseModel


class SettingsConfigDict(dict):
    """Minimal stub mirroring the behaviour expected by the app."""


class BaseSettings(_BaseModel):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover - compatibility shim
        return super().model_dump(*args, **kwargs)

    @classmethod
    def model_validate(cls, value: Any) -> "BaseSettings":  # pragma: no cover
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported value for settings validation: {type(value)!r}")


__all__ = ["BaseSettings", "SettingsConfigDict"]
