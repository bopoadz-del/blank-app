"""Minimal TestClient implementation for the FastAPI stubs."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from .app import FastAPI, Response


class TestClient:
    """Synchronous test client invoking the in-process application."""

    def __init__(self, app: FastAPI):
        self.app = app
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._enter_lifespan()

    def _enter_lifespan(self) -> None:
        try:
            self._loop.run_until_complete(self.app._enter_lifespan())
        except Exception as exc:
            logging.error(
                "Exception occurred during TestClient._enter_lifespan: %s",
                exc,
                exc_info=True,
            )

    def _exit_lifespan(self) -> None:
        try:
            self._loop.run_until_complete(self.app._exit_lifespan())
        except Exception as exc:
            logging.error(
                "Exception occurred during TestClient._exit_lifespan: %s",
                exc,
                exc_info=True,
            )
        finally:
            self._loop.close()

    def request(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None, json: Any = None, params: Optional[Dict[str, Any]] = None) -> Response:
        return self.app.handle_request(method, url, headers=headers, json=json, params=params)

    def get(self, url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Response:
        return self.request("GET", url, headers=headers, params=params)

    def post(self, url: str, *, headers: Optional[Dict[str, str]] = None, json: Any = None, params: Optional[Dict[str, Any]] = None) -> Response:
        return self.request("POST", url, headers=headers, json=json, params=params)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self._exit_lifespan()
