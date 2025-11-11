"""CORS middleware stub for compatibility."""
<<<<< codex/fix-failed-ci-and-security-scan-workflows-h3h6es
from __future__ import annotations

from stub_utils import load_actual_module

_actual_module = load_actual_module(__name__, __file__)
if _actual_module is not None:
    globals().update({name: getattr(_actual_module, name) for name in dir(_actual_module)})
    __doc__ = getattr(_actual_module, "__doc__")
    __all__ = getattr(_actual_module, "__all__", [name for name in globals() if not name.startswith("_")])
else:
    _IS_STUB_IMPLEMENTATION = True

    class CORSMiddleware:  # pragma: no cover - behavior not needed in tests
        def __init__(self, app=None, **kwargs):
            self.app = app
            self.options = kwargs

    __all__ = ["CORSMiddleware"]

__all__ = list(__all__)
=======


class CORSMiddleware:  # pragma: no cover - behavior not needed in tests
    def __init__(self, app=None, **kwargs):
        self.app = app
        self.options = kwargs
>>>>> main
