"""Test configuration ensuring offline stubs are available when dependencies are missing."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

STUB_ROOT = Path(__file__).resolve().parent / "offline_stubs"


def _ensure_dependency(module_name: str) -> None:
    """Import a dependency or fall back to the local offline stub."""
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass

    if str(STUB_ROOT) not in sys.path:
        sys.path.insert(0, str(STUB_ROOT))

    importlib.import_module(module_name)


for _module in ("fastapi", "sqlalchemy", "pydantic_settings"):
    _ensure_dependency(_module)
