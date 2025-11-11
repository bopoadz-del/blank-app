<<<<<< codex/fix-failed-ci-and-security-scan-workflows-1yqgdg
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

=======
"""Lightweight SQLAlchemy stubs for offline tests."""
from __future__ import annotations

from datetime import datetime
from typing import Any


class OperationalError(Exception):
    pass


class _DummyConnection:
    def __enter__(self) -> "_DummyConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement: Any) -> None:  # pragma: no cover - no real SQL execution
        return None


class Engine:
    def connect(self) -> _DummyConnection:
        return _DummyConnection()


def create_engine(url: str, **kwargs: Any) -> Engine:
    return Engine()


def text(sql: str) -> str:
    return sql


class Column:
    def __init__(self, column_type: Any, primary_key: bool = False, index: bool = False, nullable: bool = True, default: Any = None, server_default: Any = None):
        self.type = column_type
        self.primary_key = primary_key
        self.index = index
        self.nullable = nullable
        self.default = default
        self.server_default = server_default

    def desc(self) -> "Column":  # pragma: no cover - ordering direction hint
        return self


class Integer:
    pass


class String:
    pass


class Float:
    pass


class Boolean:
    pass


class JSON:
    pass


class DateTime:
    def __init__(self, timezone: bool = False):
        self.timezone = timezone


class _FuncModule:
    @staticmethod
    def now() -> datetime:
        return datetime.utcnow()


func = _FuncModule()


# Submodules populated below
from .ext.declarative import declarative_base  # noqa: E402  # pylint: disable=wrong-import-position
from .orm.session import sessionmaker  # noqa: E402  # pylint: disable=wrong-import-position

__all__ = [
    "create_engine",
    "text",
    "Column",
    "Integer",
    "String",
    "Float",
    "Boolean",
    "JSON",
    "DateTime",
    "func",
    "declarative_base",
    "sessionmaker",
    "OperationalError",
]
>>>>>> main
