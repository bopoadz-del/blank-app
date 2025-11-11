"""Lightweight SQLAlchemy compatibility layer for offline testing."""
from __future__ import annotations

from datetime import datetime
from typing import Any


class OperationalError(Exception):
    """Exception raised when the stubbed engine cannot complete an operation."""


class _DummyConnection:
    def __enter__(self) -> "_DummyConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement: Any) -> None:  # pragma: no cover - the stub never executes SQL
        return None


class Engine:
    """Extremely small stand-in for :class:`sqlalchemy.engine.Engine`."""

    def connect(self) -> "_DummyConnection":
        return _DummyConnection()


def create_engine(url: str, **_: Any) -> Engine:
    """Return a stubbed engine object."""

    return Engine()


def text(sql: str) -> str:
    """Represent a SQL statement string."""

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
    def __init__(self, timezone: bool = False) -> None:
        self.timezone = timezone


class _FuncModule:
    @staticmethod
    def now() -> datetime:
        return datetime.utcnow()


func = _FuncModule()

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
