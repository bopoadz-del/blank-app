"""Very small HTTP client exercising the FastAPI stub synchronously."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .app import FastAPI


class ResponseWrapper:
    def __init__(self, response):
        self.status_code = response.status_code
        self._response = response

    def json(self) -> Any:
        return self._response.json()


class TestClient:
    def __init__(self, app: FastAPI):
        self.app = app

    def get(self, path: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> ResponseWrapper:
        return ResponseWrapper(self.app.handle_request("GET", path, headers=headers, params=params))

    def post(self, path: str, json: Optional[Dict[str, Any]] = None, *, headers: Optional[Dict[str, str]] = None) -> ResponseWrapper:
        return ResponseWrapper(self.app.handle_request("POST", path, headers=headers, json=json))
