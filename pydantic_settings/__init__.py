"""Minimal subset of pydantic-settings for tests."""
from __future__ import annotations

<<<<< codex/fix-failed-ci-and-security-scan-workflows-h3h6es
from stub_utils import load_actual_module

_actual_module = load_actual_module(__name__, __file__)
if _actual_module is not None:
    globals().update({name: getattr(_actual_module, name) for name in dir(_actual_module)})
    __doc__ = getattr(_actual_module, "__doc__")
    __all__ = getattr(_actual_module, "__all__", [name for name in globals() if not name.startswith("_")])
else:
    _IS_STUB_IMPLEMENTATION = True

    from typing import Any

    from pydantic import BaseModel

    class BaseSettings(BaseModel):
        class Config:
            env_file = None
            case_sensitive = False

        def __init__(self, **values: Any):
            super().__init__(**values)

    __all__ = ["BaseSettings"]

__all__ = list(__all__)
=======
from typing import Any

from pydantic import BaseModel


class BaseSettings(BaseModel):
    class Config:
        env_file = None
        case_sensitive = False

    def __init__(self, **values: Any):
        super().__init__(**values)
>>>>> main
