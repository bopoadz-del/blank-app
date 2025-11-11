"""Declarative base stub."""
from __future__ import annotations

from typing import Any


class _Metadata:
    def create_all(self, bind: Any = None) -> None:  # pragma: no cover - metadata operations are no-ops
        return None

    def drop_all(self, bind: Any = None) -> None:  # pragma: no cover - metadata operations are no-ops
        try:
            from ..orm.session import Session

            Session._store.clear()
            Session._id_counter.clear()
        except Exception:
            # This is a stub implementation, so we intentionally ignore exceptions.
            pass


def declarative_base() -> type:
    class Base:
        metadata = _Metadata()

        def __init__(self, **kwargs: Any):
            for key, value in kwargs.items():
                setattr(self, key, value)
    return Base
