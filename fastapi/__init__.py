<<<<<< codex/fix-failed-ci-and-security-scan-workflows-m4ja6v
"""Load the real FastAPI package when available, otherwise use the lightweight shim."""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import List, Optional


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
    return importlib.import_module("app.shims.fastapi")


def _alias_submodules(source_prefix: str, target_prefix: str, modules: List[str]) -> None:
    for suffix in modules:
        source_name = f"{source_prefix}{suffix}"
        target_name = f"{target_prefix}{suffix}"
        module = importlib.import_module(source_name)
        sys.modules[target_name] = module


_module = _load_real_package()
if _module is None:
    _module = _load_shim()
    _alias_submodules(
        "app.shims.fastapi",
        __name__,
        [
            ".app",
            ".middleware",
            ".middleware.cors",
            ".security",
            ".status",
            ".testclient",
        ],
    )

sys.modules[__name__] = _module
globals().update({name: getattr(_module, name) for name in dir(_module)})
__all__ = getattr(_module, "__all__", [])
=======
"""Lightweight FastAPI-compatible stubs for offline testing."""
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
>>>>>> main
