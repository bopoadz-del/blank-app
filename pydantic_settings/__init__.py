"""Provide pydantic-settings if installed, otherwise expose the shim implementation."""

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Optional


@contextmanager
def _without_project_root() -> None:
    project_root = Path(__file__).resolve().parent.parent
    original = list(sys.path)
    try:
        sys.path = [entry for entry in original if Path(entry or ".").resolve() != project_root]
        yield
    finally:
        sys.path = original


def _load_real_package() -> Optional[ModuleType]:
    module_name = __name__
    with _without_project_root():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            return None
    if Path(getattr(module, "__file__", "")).resolve() == Path(__file__).resolve():
        return None
    return module


def _load_shim() -> ModuleType:
    return importlib.import_module("app.shims.pydantic_settings")


_module = _load_real_package() or _load_shim()

sys.modules[__name__] = _module
globals().update({name: getattr(_module, name) for name in dir(_module)})
__all__ = getattr(_module, "__all__", [])
