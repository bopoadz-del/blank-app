"""Expose the real pydantic-settings package when available, otherwise use the shim."""

import importlib
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _search_paths() -> Iterable[str]:
    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if candidate == _PROJECT_ROOT:
            continue
        yield str(candidate)


def _load_real() -> Optional[ModuleType]:
    for entry in _search_paths():
        spec = PathFinder.find_spec(__name__, [entry])
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def _load_shim() -> ModuleType:
    return importlib.import_module("app.shims.pydantic_settings")


_module = _load_real() or _load_shim()
sys.modules[__name__] = _module
globals().update({name: getattr(_module, name) for name in dir(_module)})
__all__ = getattr(_module, "__all__", [])

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
