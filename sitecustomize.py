"""Provide lightweight dependency shims during local execution.

This module is imported automatically by Python when present on the import
path. We use it to supply minimal stand-ins for heavy optional dependencies
when they are not installed (for example in offline CI environments). The
fallback modules live under :mod:`app._shims` and are only activated if the
real dependency cannot be imported.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

_STUB_ROOT = Path(__file__).resolve().parent / "app" / "_shims"


def _load_stub(module_name: str) -> None:
    """Load the shim package for *module_name* if it exists."""

    stub_init = _STUB_ROOT / module_name.replace(".", "/") / "__init__.py"
    if not stub_init.exists():  # pragma: no cover - guard for safety
        return

    spec = importlib.util.spec_from_file_location(
        module_name,
        stub_init,
        submodule_search_locations=[str(stub_init.parent)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _ensure_dependency(name: str) -> None:
    """Ensure *name* is importable, falling back to the shim when necessary."""

    if name in sys.modules:
        return

    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        _load_stub(name)


for dependency in ("fastapi", "sqlalchemy", "pydantic_settings"):
    _ensure_dependency(dependency)


def ensure_dependencies(names: Iterable[str]) -> None:
    """Public helper to ensure a collection of dependency shims is loaded."""

    for dep in names:
        _ensure_dependency(dep)
