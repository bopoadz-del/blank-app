"""Compatibility loader for SQLAlchemy.

Searches for an installed SQLAlchemy distribution outside of the project root
and falls back to the lightweight shim when none is present.
"""

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
    return importlib.import_module("app.shims.sqlalchemy")


_module = _load_real()
if _module is None:
    _module = _load_shim()
    sys.modules.setdefault(__name__ + ".exc", importlib.import_module("app.shims.sqlalchemy.exc"))
    sys.modules.setdefault(__name__ + ".ext", importlib.import_module("app.shims.sqlalchemy.ext"))
    sys.modules.setdefault(
        __name__ + ".ext.declarative",
        importlib.import_module("app.shims.sqlalchemy.ext.declarative"),
    )
    sys.modules.setdefault(__name__ + ".orm", importlib.import_module("app.shims.sqlalchemy.orm"))
    sys.modules.setdefault(
        __name__ + ".orm.session",
        importlib.import_module("app.shims.sqlalchemy.orm.session"),
    )
    sys.modules.setdefault(__name__ + ".sql", importlib.import_module("app.shims.sqlalchemy.sql"))

sys.modules[__name__] = _module
globals().update({name: getattr(_module, name) for name in dir(_module)})
__all__ = getattr(_module, "__all__", [])

