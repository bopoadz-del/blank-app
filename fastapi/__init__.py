<<<<<< codex/fix-failed-ci-and-security-scan-workflows-rvba2c
"""Load the real FastAPI package when available, otherwise use the lightweight shim."""
from __future__ import annotations

import sys
from types import ModuleType

from app.shims._proxy import load_module


def _load() -> ModuleType:
    module, _ = load_module(
        module_name=__name__,
        shim_package="app.shims.fastapi",
        module_file=__file__,
        shim_submodules=(
            ".app",
            ".middleware",
            ".middleware.cors",
            ".security",
            ".status",
            ".testclient",
        ),
    )
    return module


_module = _load()
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
