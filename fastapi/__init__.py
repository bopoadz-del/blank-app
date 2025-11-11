<<<<< codex/fix-failed-ci-and-security-scan-workflows-6yxfgs
"""Compatibility shim that prefers real FastAPI when available."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _try_load_real_package() -> ModuleType | None:
    """Attempt to import the actual FastAPI package if installed."""

    module_name = __name__
    module_self = sys.modules.get(module_name)
    project_root = Path(__file__).resolve().parent.parent
    original_sys_path = list(sys.path)

    try:
        if module_name in sys.modules:
            sys.modules.pop(module_name)

        filtered_path = []
        project_root_resolved = project_root.resolve()
        for entry in original_sys_path:
            entry_path = Path(entry or ".").resolve()
            if entry_path == project_root_resolved:
                continue
            filtered_path.append(entry)

        sys.path = filtered_path
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module = None
    finally:
        sys.path = original_sys_path
        if module_self is not None:
            sys.modules[module_name] = module_self
        else:
            sys.modules.pop(module_name, None)

    if module is not None and Path(getattr(module, "__file__", "")) == Path(__file__).resolve():
        # The import resolved back to this stub; treat as missing real package.
        return None

    return module


_real_fastapi = _try_load_real_package()

if _real_fastapi:
    sys.modules[__name__] = _real_fastapi

    special_attrs = {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__path__",
        "__file__",
        "__doc__",
        "__all__",
    }

    for attr, value in _real_fastapi.__dict__.items():
        if attr not in special_attrs:
            globals()[attr] = value

    for attr in special_attrs:
        if hasattr(_real_fastapi, attr):
            globals()[attr] = getattr(_real_fastapi, attr)

    __all__ = getattr(_real_fastapi, "__all__", [])
else:
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


del _try_load_real_package
del _real_fastapi
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
