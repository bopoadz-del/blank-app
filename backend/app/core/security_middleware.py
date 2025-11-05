"""
Security middleware for hardening the application.
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from typing import Callable
import time
import re
import logging

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp

        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize incoming requests."""

    # Patterns for common attack vectors
    SQL_INJECTION_PATTERN = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)|"
        r"(--|;|\/\*|\*\/|xp_|sp_|@|char\(|cast\()",
        re.IGNORECASE
    )

    XSS_PATTERN = re.compile(
        r"<script|javascript:|onerror=|onload=|onclick=|<iframe|<object|<embed",
        re.IGNORECASE
    )

    PATH_TRAVERSAL_PATTERN = re.compile(r"\.\./|\.\.\\")

    async def dispatch(self, request: Request, call_next: Callable):
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length:
            if int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": "Request body too large"}
                )

        # Validate query parameters
        for key, value in request.query_params.items():
            if self._is_malicious(value):
                logger.warning(f"Malicious query parameter detected: {key}={value[:50]}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid request parameters"}
                )

        # Validate path
        if self.PATH_TRAVERSAL_PATTERN.search(str(request.url.path)):
            logger.warning(f"Path traversal attempt detected: {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid request path"}
            )

        response = await call_next(request)
        return response

    def _is_malicious(self, value: str) -> bool:
        """Check if value contains malicious patterns."""
        if self.SQL_INJECTION_PATTERN.search(value):
            return True
        if self.XSS_PATTERN.search(value):
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Custom rate limiting middleware."""

    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}

    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = get_remote_address(request)

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        current_time = time.time()

        # Clean old entries
        self.clients = {
            ip: times for ip, times in self.clients.items()
            if any(t > current_time - self.period for t in times)
        }

        # Get client request times
        if client_ip not in self.clients:
            self.clients[client_ip] = []

        # Filter recent requests
        self.clients[client_ip] = [
            t for t in self.clients[client_ip]
            if t > current_time - self.period
        ]

        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": self.period
                },
                headers={"Retry-After": str(self.period)}
            )

        # Add current request
        self.clients[client_ip].append(current_time)

        response = await call_next(request)
        return response


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """IP whitelist/blacklist middleware."""

    def __init__(self, app, whitelist: list = None, blacklist: list = None):
        super().__init__(app)
        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])

    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = get_remote_address(request)

        # Check blacklist
        if client_ip in self.blacklist:
            logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"}
            )

        # Check whitelist (if configured)
        if self.whitelist and client_ip not in self.whitelist:
            logger.warning(f"Blocked request from non-whitelisted IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"}
            )

        response = await call_next(request)
        return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log all API requests for security auditing."""

    SENSITIVE_PATHS = ["/auth/login", "/auth/register", "/auth/refresh"]
    SENSITIVE_HEADERS = ["authorization", "x-api-key"]

    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()

        # Log request
        client_ip = get_remote_address(request)
        user_agent = request.headers.get("user-agent", "Unknown")

        # Sanitize headers
        headers = dict(request.headers)
        for header in self.SENSITIVE_HEADERS:
            if header in headers:
                headers[header] = "***REDACTED***"

        # Log sensitive operations
        if request.url.path in self.SENSITIVE_PATHS or request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            logger.info(
                f"[AUDIT] {request.method} {request.url.path} from {client_ip} "
                f"User-Agent: {user_agent}"
            )

        response = await call_next(request)

        # Log response time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Log failed auth attempts
        if request.url.path in self.SENSITIVE_PATHS and response.status_code >= 400:
            logger.warning(
                f"[AUDIT] Failed {request.method} {request.url.path} from {client_ip} "
                f"Status: {response.status_code}"
            )

        return response


def setup_security_middleware(app, config: dict = None):
    """Setup all security middleware."""
    config = config or {}

    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Trusted hosts (if configured)
    allowed_hosts = config.get("allowed_hosts", ["*"])
    if allowed_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Request validation
    app.add_middleware(RequestValidationMiddleware)

    # Rate limiting
    rate_limit_calls = config.get("rate_limit_calls", 100)
    rate_limit_period = config.get("rate_limit_period", 60)
    app.add_middleware(RateLimitMiddleware, calls=rate_limit_calls, period=rate_limit_period)

    # IP filtering (if configured)
    whitelist = config.get("ip_whitelist", [])
    blacklist = config.get("ip_blacklist", [])
    if whitelist or blacklist:
        app.add_middleware(IPWhitelistMiddleware, whitelist=whitelist, blacklist=blacklist)

    # Audit logging
    app.add_middleware(AuditLogMiddleware)

    # Add rate limit handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info("Security middleware configured")

    return limiter
