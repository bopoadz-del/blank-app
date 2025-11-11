"""Load SQLAlchemy from the environment with a shim fallback for offline tests."""
from __future__ import annotations

import sys
from types import ModuleType

from app.shims._proxy import load_module


def _load() -> ModuleType:
    module, _ = load_module(
        module_name=__name__,
        shim_package="app.shims.sqlalchemy",
        module_file=__file__,
        shim_submodules=(
            ".exc",
            ".ext",
            ".ext.declarative",
            ".orm",
            ".orm.session",
            ".sql",
        ),
    )
    return module


_module = _load()
sys.modules[__name__] = _module
globals().update({name: getattr(_module, name) for name in dir(_module)})
__all__ = getattr(_module, "__all__", [])
