<<<<<< codex/fix-failed-ci-and-security-scan-workflows-m4ja6v
"""Load SQLAlchemy from the environment with a shim fallback for offline tests."""
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
    return importlib.import_module("app.shims.sqlalchemy")


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
        "app.shims.sqlalchemy",
        __name__,
        [
            ".exc",
            ".ext",
            ".ext.declarative",
            ".orm",
            ".orm.session",
            ".sql",
        ],
    )

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
