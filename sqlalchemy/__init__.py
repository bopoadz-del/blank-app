"""Extremely small SQLAlchemy shim for offline tests."""
from __future__ import annotations

from stub_utils import load_actual_module

_actual_module = load_actual_module(__name__, __file__)
if _actual_module is not None:
    globals().update({name: getattr(_actual_module, name) for name in dir(_actual_module)})
    __doc__ = getattr(_actual_module, "__doc__")
    __all__ = getattr(_actual_module, "__all__", [name for name in globals() if not name.startswith("_")])
else:
    _IS_STUB_IMPLEMENTATION = True

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

    def create_engine(url: str, **kwargs: Any) -> Engine:  # pragma: no cover - url unused
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
        ):
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

__all__ = list(__all__)
