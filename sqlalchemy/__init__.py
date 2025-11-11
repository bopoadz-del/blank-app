"""Load SQLAlchemy from the environment with a shim fallback for offline tests."""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Optional
from typing import List


@contextmanager
def _without_project_root() -> None:
    project_root = Path(__file__).resolve().parent.parent
    original = list(sys.path)
    try:
        sys.path = [entry for entry in original if Path(entry or ".").resolve() != project_root]
        yield
    finally:
        sys.path = original


def _load_real_package() -> Optional[ModuleType]:
    module_name = __name__
    with _without_project_root():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            return None
    if Path(getattr(module, "__file__", "")).resolve() == Path(__file__).resolve():
        return None
    return module


def _load_shim() -> ModuleType:
    return importlib.import_module("app.shims.sqlalchemy")


def _alias_submodules(source_prefix: str, target_prefix: str, modules: List[str]) -> None:
    for suffix in modules:
        source_name = f"{source_prefix}{suffix}"
        target_name = f"{target_prefix}{suffix}"
        module = importlib.import_module(source_name)
        sys.modules[target_name] = module


_module = _load_real_package()
if _module is None:
    _module = _load_shim()
    _alias_submodules(
        "app.shims.sqlalchemy",
        __name__,
        [
            ".exc",
            ".ext",
            ".ext.declarative",
            ".orm",
            ".orm.session",
            ".sql",
        ],
    )

sys.modules[__name__] = _module
globals().update({name: getattr(_module, name) for name in dir(_module)})
__all__ = getattr(_module, "__all__", [])
