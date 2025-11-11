"""Test package initialization that installs lightweight stubs when needed."""
from __future__ import annotations

import importlib
import sys
from typing import Dict


def _install_stub(package: str, base: str, submodules: Dict[str, str]) -> None:
    stub = importlib.import_module(base)
    sys.modules[package] = stub
    for name, target in submodules.items():
        module_name = f"{base}.{target}" if target else base
        sys.modules[f"{package}.{name}" if name else package] = importlib.import_module(module_name)


def _ensure_fastapi() -> None:
    try:
        importlib.import_module("fastapi")
    except Exception:
        _install_stub(
            "fastapi",
            "tests._stubs.fastapi",
            {
                "": "",
                "app": "app",
                "middleware": "middleware",
                "middleware.cors": "middleware.cors",
                "security": "security",
                "status": "status",
                "testclient": "testclient",
            },
        )


def _ensure_sqlalchemy() -> None:
    try:
        importlib.import_module("sqlalchemy")
    except Exception:
        _install_stub(
            "sqlalchemy",
            "tests._stubs.sqlalchemy",
            {
                "": "",
                "exc": "exc",
                "ext": "ext",
                "ext.declarative": "ext.declarative",
                "orm": "orm",
                "orm.session": "orm.session",
                "sql": "sql",
            },
        )


def _ensure_pydantic_settings() -> None:
    try:
        importlib.import_module("pydantic_settings")
    except Exception:
        _install_stub(
            "pydantic_settings",
            "tests._stubs.pydantic_settings",
            {"": ""},
        )


_ensure_fastapi()
_ensure_sqlalchemy()
_ensure_pydantic_settings()
