"""Minimal FastAPI-compatible application primitives used in tests."""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from .status import status


class HTTPException(Exception):
    """Simplified HTTPException for test usage."""

    def __init__(self, status_code: int, detail: Any = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class Depends:
    dependency: Callable[..., Any]


@dataclass
class Security(Depends):
    pass


class Request:
    """Very small request object passed to handlers and dependencies."""

    def __init__(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, json: Any = None, params: Optional[Dict[str, Any]] = None):
        self.method = method.upper()
        self.url = url
        self.headers = {k.title(): v for k, v in (headers or {}).items()}
        self._json = json
        self.query_params = params or {}
        self.state: Dict[str, Any] = {}

    @property
    def client(self) -> Any:
        class _Client:
            host = "testclient"
        return _Client()

    def json(self) -> Any:
        return self._json


class _Route:
    def __init__(self, method: str, path: str, endpoint: Callable[..., Any]):
        self.method = method
        self.path = self._normalize_path(path)
        self.endpoint = endpoint
        self._segments = self.path.strip("/").split("/") if self.path != "/" else []

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return path

    def match(self, path: str) -> Optional[Dict[str, str]]:
        candidate = self._normalize_path(path)
        if self.method == "GET" and candidate == "/" and self.path == "/":
            return {}
        cand_segments = candidate.strip("/").split("/") if candidate != "/" else []
        if len(cand_segments) != len(self._segments):
            return None
        params: Dict[str, str] = {}
        for route_segment, cand_segment in zip(self._segments, cand_segments):
            if route_segment.startswith("{") and route_segment.endswith("}"):
                param_name = route_segment[1:-1]
                params[param_name] = cand_segment
            elif route_segment != cand_segment:
                return None
        return params


class APIRouter:
    """Router container mirroring the subset used in tests."""

    def __init__(self):
        self.routes: List[_Route] = []

    def get(self, path: str, response_model: Any = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add("GET", path)

    def post(self, path: str, response_model: Any = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add("POST", path)

    def _add(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes.append(_Route(method, path, func))
            return func
        return decorator


class FastAPI:
    """Tiny FastAPI replacement supporting the repository's tests."""

    def __init__(self, title: str = "FastAPI", version: str = "0.1.0", description: str | None = None, lifespan: Optional[Callable[["FastAPI"], Awaitable[Any]]] = None):
        self.title = title
        self.version = version
        self.description = description
        self._routes: List[_Route] = []
        self._middleware: List[Tuple[Any, Dict[str, Any]]] = []
        self._lifespan_factory = lifespan
        self._lifespan_context = None
        self.dependency_overrides: Dict[Callable[..., Any], Callable[..., Any]] = {}

    def add_middleware(self, middleware_cls: Any, **kwargs: Any) -> None:
        self._middleware.append((middleware_cls, kwargs))

    def include_router(self, router: APIRouter, prefix: str = "", tags: Optional[List[str]] = None) -> None:
        prefix = prefix.rstrip("/")
        for route in router.routes:
            combined_path = f"{prefix}{route.path}" if prefix else route.path
            self._routes.append(_Route(route.method, combined_path, route.endpoint))

    def get(self, path: str, response_model: Any = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add("GET", path)

    def post(self, path: str, response_model: Any = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add("POST", path)

    def _add(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes.append(_Route(method, path, func))
            return func
        return decorator

    async def _enter_lifespan(self) -> None:
        if self._lifespan_factory is None:
            return
        self._lifespan_context = self._lifespan_factory(self)
        await self._lifespan_context.__aenter__()

    async def _exit_lifespan(self) -> None:
        if self._lifespan_context is None:
            return
        await self._lifespan_context.__aexit__(None, None, None)
        self._lifespan_context = None

    def find_route(self, method: str, path: str) -> Tuple[Optional[_Route], Dict[str, str]]:
        for route in self._routes:
            if route.method != method.upper():
                continue
            params = route.match(path)
            if params is not None:
                return route, params
        return None, {}

    def handle_request(self, method: str, path: str, *, headers: Optional[Dict[str, str]] = None, json: Any = None, params: Optional[Dict[str, Any]] = None) -> "Response":
        route, path_params = self.find_route(method, path)
        if not route:
            return Response(status.HTTP_404_NOT_FOUND, {"detail": "Not Found"})

        request = Request(method, path, headers=headers, json=json, params=params)
        try:
            result = self._invoke_endpoint(route.endpoint, request, path_params)
            status_code = status.HTTP_200_OK
            body = result
            if isinstance(result, tuple) and len(result) == 2:
                body, status_code = result
        except HTTPException as exc:  # pragma: no cover - handled explicitly in tests
            status_code = exc.status_code
            body = {"detail": exc.detail}
        return Response(status_code, self._prepare_body(body))

    def _prepare_body(self, body: Any) -> Any:
        if body is None:
            return None
        if isinstance(body, BaseModel):
            return body.model_dump()
        if hasattr(body, "model_dump"):
            return body.model_dump()
        return body

    def _invoke_endpoint(self, endpoint: Callable[..., Any], request: Request, path_params: Dict[str, str]) -> Any:
        signature = inspect.signature(endpoint)
        kwargs: Dict[str, Any] = {}
        cleanups: List[Callable[[], None]] = []

        try:
            for name, parameter in signature.parameters.items():
                if name == "request":
                    kwargs[name] = request
                    continue

                default = parameter.default
                if isinstance(default, Depends):
                    value, cleanup = self._resolve_dependency(default.dependency, request)
                    kwargs[name] = value
                    if cleanup:
                        cleanups.append(cleanup)
                    continue

                if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    assigned = False
                    if name in path_params:
                        kwargs[name] = self._coerce_value(path_params[name], parameter.annotation)
                        assigned = True
                    if parameter.annotation and isinstance(request.json(), dict):
                        json_payload = request.json()
                        if inspect.isclass(parameter.annotation) and issubclass(parameter.annotation, BaseModel):
                            kwargs[name] = parameter.annotation(**json_payload)
                            assigned = True
                        elif name in json_payload:
                            kwargs[name] = json_payload[name]
                            assigned = True
                    if not assigned and name in request.query_params:
                        raw_value = request.query_params[name]
                        kwargs[name] = self._coerce_value(raw_value, parameter.annotation)
                        assigned = True
                    if not assigned:
                        kwargs[name] = None

            result = self._call(endpoint, kwargs)
        finally:
            for cleanup in reversed(cleanups):
                cleanup()
        return result

    def _resolve_dependency(self, dependency: Callable[..., Any], request: Request) -> Tuple[Any, Optional[Callable[[], None]]]:
        dependency = self.dependency_overrides.get(dependency, dependency)
        if isinstance(dependency, APIKeyDependency):
            return dependency(request), None

        if inspect.isgeneratorfunction(dependency):
            generator = dependency()
            value = next(generator)

            def _cleanup() -> None:
                try:
                    generator.close()
                except Exception:
                    logging.exception("Exception ignored while closing generator in dependency cleanup.")

            return value, _cleanup

        signature = inspect.signature(dependency)
        kwargs: Dict[str, Any] = {}
        cleanups: List[Callable[[], None]] = []
        for name, parameter in signature.parameters.items():
            default = parameter.default
            if name == "request":
                kwargs[name] = request
            elif isinstance(default, Depends):
                value, cleanup = self._resolve_dependency(default.dependency, request)
                kwargs[name] = value
                if cleanup:
                    cleanups.append(cleanup)
        result = self._call(dependency, kwargs)
        cleanup_fn: Optional[Callable[[], None]] = None
        if cleanups:
            def combined_cleanup() -> None:
                for func in reversed(cleanups):
                    func()
            cleanup_fn = combined_cleanup
        return result, cleanup_fn

    def _call(self, func: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(func):
            return asyncio.run(func(**kwargs))
        return func(**kwargs)

    @staticmethod
    def _coerce_value(value: Any, annotation: Any) -> Any:
        if annotation in (int, float, bool, str):
            try:
                if annotation is bool:
                    return str(value).lower() in {"1", "true", "yes", "on"}
                return annotation(value)
            except Exception:
                return value
        return value

class APIKeyDependency:
    """Callable wrapper implementing header-based API key retrieval."""

    def __init__(self, name: str, auto_error: bool = True):
        self.name = name.title()
        self.auto_error = auto_error

    def __call__(self, request: Request) -> Optional[str]:
        value = request.headers.get(self.name)
        if value is None and self.auto_error:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not authenticated")
        return value


class Response:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


# Re-export for convenience in other modules inside the stub package
APIKeyHeader = APIKeyDependency
