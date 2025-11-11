"""Test stub emulating a subset of SQLAlchemy."""
from __future__ import annotations

from typing import Any

from ._support import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    JSON,
    MetaData,
    String,
    func,
)
from .exc import OperationalError
from .orm.session import Session, sessionmaker
from .ext.declarative import declarative_base
from .sql import text


def create_engine(url: str, **kwargs: Any) -> Engine:
    return Engine(url, **kwargs)


__all__ = [
    "Boolean",
    "Column",
    "DateTime",
    "Engine",
    "Float",
    "Integer",
    "JSON",
    "MetaData",
    "OperationalError",
    "String",
    "Session",
    "create_engine",
    "declarative_base",
    "func",
    "sessionmaker",
    "text",
]
