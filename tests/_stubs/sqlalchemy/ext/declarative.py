"""Declarative mapping for the SQLAlchemy stub."""
from __future__ import annotations

from typing import Any, Dict

from .._support import Column, MetaData, evaluate_default


class DeclarativeMeta(type):
    def __new__(mcls, name: str, bases: tuple[type, ...], attrs: Dict[str, Any]):
        columns = {key: value for key, value in list(attrs.items()) if isinstance(value, Column)}
        for key, column in columns.items():
            attrs[key] = column.create_accessor(key)

        metadata = attrs.get("metadata")
        if metadata is None:
            for base in bases:
                base_metadata = getattr(base, "metadata", None)
                if isinstance(base_metadata, MetaData):
                    metadata = base_metadata
                    break

        cls = super().__new__(mcls, name, bases, attrs)
        cls.__columns__ = columns
        if metadata is None:
            metadata = MetaData()
        cls.metadata = metadata
        if name != "DeclarativeBase":
            cls.metadata.register(cls)
        if "__tablename__" not in attrs:
            cls.__tablename__ = name.lower()
        return cls


def declarative_base():
    metadata = MetaData()

    class DeclarativeBase(metaclass=DeclarativeMeta):

        def __init__(self, **kwargs: Any):
            columns: Dict[str, Column] = getattr(self.__class__, "__columns__", {})
            for name, column in columns.items():
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                else:
                    default = column.default if column.default is not None else column.server_default
                    setattr(self, name, evaluate_default(default))
            for key, value in kwargs.items():
                if key not in columns:
                    setattr(self, key, value)

    DeclarativeBase.metadata = metadata

    return DeclarativeBase
