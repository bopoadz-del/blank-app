<<<<<< codex/fix-failed-ci-and-security-scan-workflows-rvba2c
"""Provide pydantic-settings if installed, otherwise expose the shim implementation."""
from __future__ import annotations

import sys
from types import ModuleType

from app.shims._proxy import load_module


def _load() -> ModuleType:
    module, _ = load_module(
        module_name=__name__,
        shim_package="app.shims.pydantic_settings",
        module_file=__file__,
    )
    return module


_module = _load()
sys.modules[__name__] = _module
globals().update({name: getattr(_module, name) for name in dir(_module)})
__all__ = getattr(_module, "__all__", [])
=======
"""Minimal subset of pydantic-settings for tests."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BaseSettings(BaseModel):
    class Config:
        env_file = None
        case_sensitive = False

    def __init__(self, **values: Any):
        super().__init__(**values)
>>>>>> main
