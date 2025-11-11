<<<<< codex/fix-failed-ci-and-security-scan-workflows-g60q29
"""Compatibility shim that prefers real pydantic-settings when available."""
from __future__ import annotations

import importlib
import sys
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


_real_settings = _try_load_real_package()

if _real_settings:
    sys.modules[__name__] = _real_settings
    globals().update({name: getattr(_real_settings, name) for name in dir(_real_settings)})
    __all__ = getattr(_real_settings, "__all__", [])
else:
    from pydantic import BaseModel

    class BaseSettings(BaseModel):
        class Config:
            env_file = None
            case_sensitive = False

        def __init__(self, **values: Any):
            super().__init__(**values)

    __all__ = ["BaseSettings"]


del _try_load_real_package
del _real_settings
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
>>>>> main
