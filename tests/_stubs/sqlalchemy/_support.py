"""Shared helpers for the SQLAlchemy stub used in tests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class OrderBy:
    column_name: str
    descending: bool = False


class ColumnAccessor:
    """Lightweight descriptor representing a mapped column."""

    def __init__(self, column_name: str):
        self.column_name = column_name

    def desc(self) -> OrderBy:
        return OrderBy(self.column_name, True)


class Column:
    """Placeholder for SQLAlchemy's Column object."""

    def __init__(
        self,
        column_type: Any,
        primary_key: bool = False,
        default: Any = None,
        nullable: bool = True,
        index: bool = False,
        server_default: Any = None,
    ) -> None:
        self.type = column_type
        self.primary_key = primary_key
        self.default = default
        self.nullable = nullable
        self.index = index
        self.server_default = server_default

    def create_accessor(self, column_name: str) -> ColumnAccessor:
        return ColumnAccessor(column_name)


class MetaData:
    """Tracks declarative models for create_all/drop_all operations."""

    def __init__(self) -> None:
        self._models: List[type] = []

    def register(self, model: type) -> None:
        if model not in self._models:
            self._models.append(model)

    def create_all(self, bind: "Engine") -> None:
        for model in self._models:
            bind._storage.setdefault(model.__tablename__, [])

    def drop_all(self, bind: "Engine") -> None:
        for model in self._models:
            bind._storage.pop(model.__tablename__, None)


class Engine:
    """Simplified engine that stores rows in memory."""

    def __init__(self, url: str, **kwargs: Any) -> None:
        self.url = url
        self.options = kwargs
        self._storage: Dict[str, List[Any]] = {}

    def connect(self) -> "Connection":
        return Connection(self)


class Connection:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def __enter__(self) -> "Connection":
        if self.engine.url.startswith("postgresql"):
            from .exc import OperationalError

            raise OperationalError("Could not connect", self.engine.url, None)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return False

    def execute(self, statement: Any) -> None:
        return None


def evaluate_default(value: Any) -> Any:
    if callable(value):
        return value()
    return value


class SQLType:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"SQLType({self.name})"

    def __call__(self, *args: Any, **kwargs: Any) -> "SQLType":
        return self


Integer = SQLType("Integer")
String = SQLType("String")
Float = SQLType("Float")
Boolean = SQLType("Boolean")
JSON = SQLType("JSON")
DateTime = SQLType("DateTime")


class FuncNamespace:
    @staticmethod
    def now() -> datetime:
        return datetime.utcnow()


func = FuncNamespace()
