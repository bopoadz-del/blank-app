"""Lightweight subset of Pydantic used by the test suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


_MISSING = object()


@dataclass
class _FieldInfo:
    default: Any = _MISSING
    default_factory: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


def Field(
    default: Any = _MISSING,
    *,
    default_factory: Optional[Any] = None,
    **kwargs: Any,
) -> _FieldInfo:
    """Return metadata for a field definition."""

    return _FieldInfo(default=default, default_factory=default_factory, metadata=kwargs)


class BaseModel:
    """Very small stand-in replicating the portions of BaseModel that tests use."""

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self.__class__, "__annotations__", {})
        for name in annotations:
            if name in data:
                value = data[name]
            else:
                value = self._resolve_default(name)
            setattr(self, name, value)

        extras = {k: v for k, v in data.items() if k not in annotations}
        for key, value in extras.items():
            setattr(self, key, value)

    @classmethod
    def _resolve_default(cls, name: str) -> Any:
        candidate = getattr(cls, name, _MISSING)
        if isinstance(candidate, _FieldInfo):
            if candidate.default is not _MISSING and candidate.default is not ...:
                return candidate.default
            if candidate.default_factory is not None:
                return candidate.default_factory()
            if candidate.default is ...:
                raise ValueError(f"Field '{name}' is required")
            return None
        if candidate is not _MISSING:
            return candidate
        return None

    def model_dump(self) -> Dict[str, Any]:
        annotations = getattr(self.__class__, "__annotations__", {})
        result = {name: getattr(self, name) for name in annotations}
        extras = {
            key: value
            for key, value in self.__dict__.items()
            if key not in annotations
        }
        result.update(extras)
        return result


__all__ = ["BaseModel", "Field"]
