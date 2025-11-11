from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Type


def _looks_like_column(value: Any) -> bool:
    return hasattr(value, "primary_key") or hasattr(value, "nullable")


class _Query:
    def __init__(self, data: List[Any]):
        self._data = data
        self._limit = None
        self._filters: List[Dict[str, Any]] = []

    def order_by(self, *args: Any, **kwargs: Any) -> "_Query":
        return self

    def limit(self, limit: int) -> "_Query":
        self._limit = limit
        return self

    def filter_by(self, **kwargs: Any) -> "_Query":
        self._filters.append(kwargs)
        return self

    def all(self) -> List[Any]:
        results = list(self._data)
        for conditions in self._filters:
            results = [item for item in results if all(getattr(item, key, None) == value for key, value in conditions.items())]
        if self._limit is not None:
            results = results[: self._limit]
        return results

    def first(self) -> Any:
        results = self.all()
        return results[0] if results else None

    def count(self) -> int:
        return len(self.all())


class Session:
    _store: Dict[Type[Any], List[Any]] = {}
    _id_counter: Dict[Type[Any], int] = {}

    def add(self, instance: Any) -> None:
        cls = type(instance)
        bucket = self._store.setdefault(cls, [])
        created_at = getattr(instance, "created_at", None)
        if _looks_like_column(created_at) or created_at is None:
            setattr(instance, "created_at", datetime.utcnow())
        identifier = getattr(instance, "id", None)
        if _looks_like_column(identifier) or identifier is None:
            next_id = self._id_counter.get(cls, 0) + 1
            self._id_counter[cls] = next_id
            setattr(instance, "id", next_id)
        bucket.append(instance)

    def commit(self) -> None:  # pragma: no cover - nothing to persist
        return None

    def refresh(self, instance: Any) -> None:  # pragma: no cover - attributes already set
        return None

    def close(self) -> None:  # pragma: no cover - nothing to cleanup
        return None

    def query(self, model: Type[Any]) -> _Query:
        bucket = self._store.setdefault(model, [])
        return _Query(bucket)


def sessionmaker(*args: Any, **kwargs: Any):
    def factory() -> Session:
        return Session()

    return factory
