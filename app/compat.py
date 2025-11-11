"""Helpers for installing lightweight compatibility shims."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional


class _StubModuleFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that serves modules from ``app/_stubs`` when needed."""

    def __init__(self, stub_root: Path) -> None:
        self._stub_root = stub_root

    def find_spec(
        self, fullname: str, path: Optional[list[str]] = None, target: Optional[ModuleType] = None
    ) -> Optional[importlib.machinery.ModuleSpec]:
        stub_location = self._stub_root.joinpath(*fullname.split("."))
        package_init = stub_location / "__init__.py"
        if package_init.exists():
            return importlib.util.spec_from_file_location(
                fullname,
                package_init,
                submodule_search_locations=[str(stub_location)],
            )

        module_file = stub_location.with_suffix(".py")
        if module_file.exists():
            return importlib.util.spec_from_file_location(fullname, module_file)

        return None


_unused_finder: Optional[_StubModuleFinder] = None


def install_shims() -> None:
    """Attach the stub finder to ``sys.meta_path`` once per interpreter."""

    global _unused_finder

    if _unused_finder is not None:
        return

    stub_root = Path(__file__).resolve().parent / "_stubs"
    if not stub_root.exists():
        return

    finder = _StubModuleFinder(stub_root)
    sys.meta_path.append(finder)
    _unused_finder = finder


__all__ = ["install_shims"]
