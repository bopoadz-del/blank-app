<<<<<< codex/fix-failed-ci-and-security-scan-workflows-rvba2c
"""Load SQLAlchemy from the environment with a shim fallback for offline tests."""
from __future__ import annotations

import sys
from types import ModuleType

from app.shims._proxy import load_module


def _load() -> ModuleType:
    module, _ = load_module(
        module_name=__name__,
        shim_package="app.shims.sqlalchemy",
        module_file=__file__,
        shim_submodules=(
            ".exc",
            ".ext",
            ".ext.declarative",
            ".orm",
            ".orm.session",
            ".sql",
        ),
    )
    return module


_module = _load()
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
