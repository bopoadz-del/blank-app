"""Declarative base stub."""
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

    class _Metadata:
        def create_all(self, bind: Any = None) -> None:  # pragma: no cover - metadata operations are no-ops
            return None

        def drop_all(self, bind: Any = None) -> None:  # pragma: no cover - metadata operations are no-ops
            try:
                from ..orm.session import Session

                Session._store.clear()
                Session._id_counter.clear()
            except Exception:
                pass

    def declarative_base() -> type:
        class Base:
            metadata = _Metadata()

            def __init__(self, **kwargs: Any):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return Base

    __all__ = ["declarative_base"]

__all__ = list(__all__)
=======
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
            pass


def declarative_base() -> type:
    class Base:
        metadata = _Metadata()

        def __init__(self, **kwargs: Any):
            for key, value in kwargs.items():
                setattr(self, key, value)
    return Base
>>>>> main
