"""Session implementation for the SQLAlchemy stub."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from .._support import Column, OrderBy, evaluate_default


class Session:
    def __init__(self, engine) -> None:
        self._engine = engine
        self._storage = engine._storage

    def add(self, obj: Any) -> None:
        table = obj.__class__.__tablename__
        store = self._storage.setdefault(table, [])
        columns: Dict[str, Column] = getattr(obj.__class__, "__columns__", {})
        for name, column in columns.items():
            current = getattr(obj, name, None)
            if current is None:
                if column.primary_key:
                    setattr(obj, name, len(store) + 1)
                elif column.server_default is not None:
                    setattr(obj, name, evaluate_default(column.server_default))
                elif column.default is not None:
                    setattr(obj, name, evaluate_default(column.default))
        store.append(obj)

    def commit(self) -> None:
        return None

    def refresh(self, obj: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, model: Type[Any]) -> "Query":
        data = list(self._storage.get(model.__tablename__, []))
        return Query(data)


@dataclass
class Query:
    data: List[Any]

    def filter_by(self, **kwargs: Any) -> "Query":
        filtered = [item for item in self.data if all(getattr(item, key, None) == value for key, value in kwargs.items())]
        return Query(filtered)

    def order_by(self, order: Any) -> "Query":
        if isinstance(order, OrderBy):
            sorted_data = sorted(self.data, key=lambda item: getattr(item, order.column_name, None), reverse=order.descending)
            return Query(sorted_data)
        return self

    def limit(self, count: int) -> "Query":
        return Query(self.data[:count])

    def all(self) -> List[Any]:
        return list(self.data)

    def first(self) -> Optional[Any]:
        return self.data[0] if self.data else None

    def count(self) -> int:
        return len(self.data)


def sessionmaker(*, bind=None, **kwargs: Any):
    def _factory() -> Session:
        return Session(bind)

    return _factory
