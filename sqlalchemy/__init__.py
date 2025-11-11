"""Compatibility shim that prefers real SQLAlchemy when available."""
from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any


def _try_load_real_package() -> ModuleType | None:
    module_name = __name__
    module_self = sys.modules.get(module_name)
    project_root = Path(__file__).resolve().parent.parent
    original_sys_path = list(sys.path)

    try:
        if module_name in sys.modules:
            sys.modules.pop(module_name)

        filtered_path = []
        project_root_resolved = project_root.resolve()
        for entry in original_sys_path:
            entry_path = Path(entry or ".").resolve()
            if entry_path == project_root_resolved:
                continue
            filtered_path.append(entry)

        sys.path = filtered_path
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module = None
    finally:
        sys.path = original_sys_path
        if module_self is not None:
            sys.modules[module_name] = module_self
        else:
            sys.modules.pop(module_name, None)

    if module is not None and Path(getattr(module, "__file__", "")) == Path(__file__).resolve():
        return None

    return module


_real_sqlalchemy = _try_load_real_package()

if _real_sqlalchemy:
    sys.modules[__name__] = _real_sqlalchemy
    globals().update({name: getattr(_real_sqlalchemy, name) for name in dir(_real_sqlalchemy)})
    __all__ = getattr(_real_sqlalchemy, "__all__", [])
else:
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


del _try_load_real_package
del _real_sqlalchemy
