<<<<< codex/fix-failed-ci-and-security-scan-workflows-zu1wmc
"""Proxy module that prefers the real FastAPI package when available."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STUB_ROOT = _PROJECT_ROOT / "app" / "_shims"


def _import_real() -> ModuleType | None:
    module_name = __name__
    original_sys_path = list(sys.path)

    try:
        sys.modules.pop(module_name, None)
        filtered_path = []
        project_root_resolved = _PROJECT_ROOT.resolve()
        for entry in original_sys_path:
            entry_path = Path(entry or ".").resolve()
            if entry_path == project_root_resolved:
                continue
            filtered_path.append(entry)

        sys.path = filtered_path
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    finally:
        sys.path = original_sys_path


def _load_stub() -> ModuleType:
    module_name = __name__
    stub_package = _STUB_ROOT / module_name.replace(".", "/")
    spec = importlib.util.spec_from_file_location(
        module_name,
        stub_package / "__init__.py",
        submodule_search_locations=[str(stub_package)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError(module_name)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Register an alias so the shim can still be imported from its original path if needed.
    sys.modules.setdefault(f"app._shims.{module_name}", module)
    spec.loader.exec_module(module)
    return module


_real_fastapi = _import_real()

if _real_fastapi is not None:
    sys.modules[__name__] = _real_fastapi
    globals().update({name: getattr(_real_fastapi, name) for name in dir(_real_fastapi)})
    __all__ = getattr(_real_fastapi, "__all__", [])
else:
    _fastapi_stub = _load_stub()
    sys.modules[__name__] = _fastapi_stub
    globals().update({name: getattr(_fastapi_stub, name) for name in dir(_fastapi_stub)})
    __all__ = getattr(_fastapi_stub, "__all__", [])


del _import_real
del _load_stub
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
