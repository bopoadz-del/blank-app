"""Minimal subset of SQLAlchemy functionality for offline execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .ext.declarative import declarative_base  # noqa: F401
from .orm.session import sessionmaker  # noqa: F401


class OperationalError(Exception):
    """Compatibility exception placeholder."""


class _DummyConnection:
    def __enter__(self) -> "_DummyConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None

    def execute(self, statement: Any) -> None:  # pragma: no cover - no SQL execution
        return None


class Engine:
    def connect(self) -> "_DummyConnection":
        return _DummyConnection()


def create_engine(url: str, **kwargs: Any) -> Engine:
    return Engine()


def text(sql: str) -> str:
    return sql


class Column:
    def __init__(
        self,
        column_type: Any,
        primary_key: bool = False,
        index: bool = False,
        nullable: bool = True,
        default: Any = None,
        server_default: Any = None,
    ) -> None:
        self.type = column_type
        self.primary_key = primary_key
        self.index = index
        self.nullable = nullable
        self.default = default
        self.server_default = server_default

    def desc(self) -> "Column":  # pragma: no cover - ordering hint only
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
