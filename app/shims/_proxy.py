"""Utilities for proxy modules that prefer real packages over local shims."""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterable, Iterator, Optional, Sequence, Tuple


@contextmanager
def _without_project_root(module_file: Path) -> Iterator[None]:
    project_root = module_file.resolve().parent.parent
    original_path = list(sys.path)
    try:
        sys.path = [
            entry
            for entry in original_path
            if Path(entry or ".").resolve() != project_root
        ]
        yield
    finally:
        sys.path = original_path


def _import_real_module(module_name: str, module_file: Path) -> Optional[ModuleType]:
    existing_module = sys.modules.pop(module_name, None)
    try:
        with _without_project_root(module_file):
            module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        if existing_module is not None:
            sys.modules[module_name] = existing_module
        return None

    module_path = Path(getattr(module, "__file__", ""))
    if not module_path or module_path.resolve() == module_file.resolve():
        if existing_module is not None:
            sys.modules[module_name] = existing_module
        return None

    return module


def _alias_submodules(
    shim_package: str, module_name: str, submodules: Sequence[str]
) -> None:
    for suffix in submodules:
        source_name = f"{shim_package}{suffix}"
        target_name = f"{module_name}{suffix}"
        sys.modules[target_name] = importlib.import_module(source_name)


def load_module(
    module_name: str,
    shim_package: str,
    module_file: str,
    shim_submodules: Iterable[str] = (),
) -> Tuple[ModuleType, bool]:
    file_path = Path(module_file)
    real_module = _import_real_module(module_name, file_path)
    if real_module is not None:
        return real_module, False

    module = importlib.import_module(shim_package)
    _alias_submodules(shim_package, module_name, list(shim_submodules))
    sys.modules[module_name] = module
    return module, True


__all__ = ["load_module"]
