<<<<< codex/fix-failed-ci-and-security-scan-workflows-h3h6es
"""FastAPI compatibility layer used for offline testing environments."""
from __future__ import annotations

from stub_utils import load_actual_module

_actual_module = load_actual_module(__name__, __file__)
if _actual_module is not None:
    globals().update({name: getattr(_actual_module, name) for name in dir(_actual_module)})
    __doc__ = getattr(_actual_module, "__doc__")
    __all__ = getattr(
        _actual_module,
        "__all__",
        [name for name in globals() if not name.startswith("_")],
    )
    __path__ = getattr(_actual_module, "__path__", [])
    __spec__ = getattr(_actual_module, "__spec__")
else:
    _IS_STUB_IMPLEMENTATION = True
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

__all__ = list(__all__)
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
>>>>> main
