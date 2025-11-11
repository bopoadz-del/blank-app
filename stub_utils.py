"""Helpers for dynamically preferring real third-party packages when available."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional

_ACTUAL_CACHE: Dict[str, ModuleType] = {}


def _resolve_path(path: str) -> Optional[Path]:
    if path is None:
        return None
    try:
        return Path(path).resolve()
    except (FileNotFoundError, RuntimeError, OSError):
        return None


def load_actual_module(module_name: str, current_file: str) -> Optional[ModuleType]:
    """Attempt to import the real module, bypassing the repository stubs."""
    cached = _ACTUAL_CACHE.get(module_name)
    if cached is not None:
        return cached

    project_root = Path(current_file).resolve().parents[1]
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        is_local_stub = False
        if existing_file is not None:
            try:
                existing_path = Path(existing_file).resolve()
            except (FileNotFoundError, RuntimeError, OSError):
                existing_path = None
            else:
                is_local_stub = existing_path is not None and project_root in existing_path.parents
        if not is_local_stub and not getattr(existing, "_IS_STUB_IMPLEMENTATION", False):
            _ACTUAL_CACHE[module_name] = existing
            return existing

    search_paths = []
    for entry in sys.path:
        resolved = _resolve_path(entry or ".")
        if resolved is None or resolved == project_root:
            continue
        search_paths.append(str(resolved))

    spec = PathFinder.find_spec(module_name, search_paths)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    original = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Remove partially initialised module so subsequent imports can try the
        # stub implementation instead of a broken state.
        if original is not None:
            sys.modules[module_name] = original
        else:
            sys.modules.pop(module_name, None)
        return None

    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        try:
            resolved_module = Path(module_file).resolve()
        except (FileNotFoundError, RuntimeError, OSError):
            resolved_module = None
        else:
            if resolved_module is not None and project_root in resolved_module.parents:
                # The module we just imported still resolves to the repository stubs,
                # so treat the import as unavailable and fall back to the lightweight
                # implementation defined locally.
                if original is not None:
                    sys.modules[module_name] = original
                else:
                    sys.modules.pop(module_name, None)
                return None

    _ACTUAL_CACHE[module_name] = module
    if original is not None:
        sys.modules[module_name] = original
    else:
        sys.modules.pop(module_name, None)
    return module
