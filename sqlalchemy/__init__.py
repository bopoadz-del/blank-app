"""Proxy module that prefers the real SQLAlchemy package when available."""
"""Proxy module that prefers the real SQLAlchemy package when available."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STUB_ROOT = _PROJECT_ROOT / "app" / "_shims"


def _import_real() -> ModuleType | None:
    module_name = __name__
    original_sys_path = list(sys.path)

    try:
        sys.modules.pop(module_name, None)
        filtered_path = []
        project_root_resolved = _PROJECT_ROOT.resolve()
        for entry in original_sys_path:
            entry_path = Path(entry or ".").resolve()
            if entry_path == project_root_resolved:
                continue
            filtered_path.append(entry)

        sys.path = filtered_path
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    finally:
        sys.path = original_sys_path


def _load_stub() -> ModuleType:
    module_name = __name__
    stub_package = _STUB_ROOT / module_name.replace(".", "/")
    spec = importlib.util.spec_from_file_location(
        module_name,
        stub_package / "__init__.py",
        submodule_search_locations=[str(stub_package)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError(module_name)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.modules.setdefault(f"app._shims.{module_name}", module)
    spec.loader.exec_module(module)
    return module


_real_sqlalchemy = _import_real()

if _real_sqlalchemy is not None:
    sys.modules[__name__] = _real_sqlalchemy
    globals().update({name: getattr(_real_sqlalchemy, name) for name in dir(_real_sqlalchemy)})
    __all__ = getattr(_real_sqlalchemy, "__all__", [])
else:
    _sqlalchemy_stub = _load_stub()
    sys.modules[__name__] = _sqlalchemy_stub
    globals().update({name: getattr(_sqlalchemy_stub, name) for name in dir(_sqlalchemy_stub)})
    __all__ = getattr(_sqlalchemy_stub, "__all__", [])


del _import_real
del _load_stub

"""Lightweight SQLAlchemy stubs for offline tests."""
from __future__ import annotations

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


def create_engine(url: str, **kwargs: Any) -> Engine:
    return Engine()


def text(sql: str) -> str:
    return sql


class Column:
    def __init__(self, column_type: Any, primary_key: bool = False, index: bool = False, nullable: bool = True, default: Any = None, server_default: Any = None):
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


# Submodules populated below
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

