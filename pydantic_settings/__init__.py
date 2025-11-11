"""Proxy module that prefers the real pydantic-settings package when available."""
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


_real_settings = _import_real()

if _real_settings is not None:
    sys.modules[__name__] = _real_settings
    globals().update({name: getattr(_real_settings, name) for name in dir(_real_settings)})
    __all__ = getattr(_real_settings, "__all__", [])
else:
    _settings_stub = _load_stub()
    sys.modules[__name__] = _settings_stub
    globals().update({name: getattr(_settings_stub, name) for name in dir(_settings_stub)})
    __all__ = getattr(_settings_stub, "__all__", [])


del _import_real
del _load_stub
