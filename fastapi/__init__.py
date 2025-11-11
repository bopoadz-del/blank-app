"""Fallback FastAPI loader.

The offline test environment for this project does not have FastAPI available,
so we provide a tiny shim under :mod:`app.shims.fastapi`.  When the real
framework is installed (for example in production or security scans) this
module defers to it by searching the rest of ``sys.path`` for another
``fastapi`` distribution and loading it directly.  When the real package is
missing we expose the shim instead.
"""

import importlib
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _search_paths() -> Iterable[str]:
    """Return sys.path entries excluding the project root."""

    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if candidate == _PROJECT_ROOT:
            continue
        yield str(candidate)


def _load_real() -> Optional[ModuleType]:
    """Load the upstream FastAPI package if one exists."""

    for entry in _search_paths():
        spec = PathFinder.find_spec(__name__, [entry])
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def _load_shim() -> ModuleType:
    """Import the lightweight compatibility shim."""

    return importlib.import_module("app.shims.fastapi")


_module = _load_real()
if _module is None:
    _module = _load_shim()
    # Ensure nested imports resolve to the shim package hierarchy
    sys.modules.setdefault(__name__ + ".middleware", importlib.import_module("app.shims.fastapi.middleware"))
    sys.modules.setdefault(__name__ + ".middleware.cors", importlib.import_module("app.shims.fastapi.middleware.cors"))
    sys.modules.setdefault(__name__ + ".security", importlib.import_module("app.shims.fastapi.security"))
    sys.modules.setdefault(__name__ + ".testclient", importlib.import_module("app.shims.fastapi.testclient"))

sys.modules[__name__] = _module
globals().update({name: getattr(_module, name) for name in dir(_module)})
__all__ = getattr(_module, "__all__", [])

