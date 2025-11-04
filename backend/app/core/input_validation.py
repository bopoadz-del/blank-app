"""
Input validation and sanitization utilities.
"""
import re
import html
from typing import Any, Dict, List, Optional
import bleach
from fastapi import HTTPException, status


class InputValidator:
    """Comprehensive input validation."""

    # Regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,50}$')
    PHONE_PATTERN = re.compile(r'^\+?1?\d{9,15}$')
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|\/\*|\*\/)",
        r"(xp_|sp_|@variable)",
        r"(char\(|cast\(|convert\()",
        r"(;.*--)",
    ]

    XSS_PATTERNS = [
        r"<script[\s\S]*?>[\s\S]*?</script>",
        r"javascript:",
        r"onerror\s*=",
        r"onload\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        if not email or len(email) > 254:
            return False
        return bool(InputValidator.EMAIL_PATTERN.match(email))

    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format."""
        if not username:
            return False
        return bool(InputValidator.USERNAME_PATTERN.match(username))

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        if not url:
            return False
        return bool(InputValidator.URL_PATTERN.match(url))

    @staticmethod
    def validate_password(password: str, min_length: int = 8) -> tuple[bool, str]:
        """
        Validate password strength.
        Returns (is_valid, error_message)
        """
        if not password:
            return False, "Password is required"

        if len(password) < min_length:
            return False, f"Password must be at least {min_length} characters"

        if len(password) > 128:
            return False, "Password is too long"

        # Check for uppercase
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"

        # Check for lowercase
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"

        # Check for digit
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"

        # Check for special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"

        return True, ""

    @staticmethod
    def sanitize_html(text: str, allowed_tags: Optional[List[str]] = None) -> str:
        """Sanitize HTML input."""
        if not text:
            return ""

        if allowed_tags is None:
            allowed_tags = []

        # Use bleach to clean HTML
        return bleach.clean(
            text,
            tags=allowed_tags,
            strip=True
        )

    @staticmethod
    def escape_html(text: str) -> str:
        """Escape HTML entities."""
        if not text:
            return ""
        return html.escape(text)

    @staticmethod
    def check_sql_injection(text: str) -> bool:
        """Check if text contains SQL injection patterns."""
        if not text:
            return False

        text_lower = text.lower()
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def check_xss(text: str) -> bool:
        """Check if text contains XSS patterns."""
        if not text:
            return False

        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def validate_and_sanitize(
        text: str,
        max_length: Optional[int] = None,
        allow_html: bool = False
    ) -> str:
        """Validate and sanitize text input."""
        if not text:
            return ""

        # Check length
        if max_length and len(text) > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input too long. Maximum length is {max_length}"
            )

        # Check for SQL injection
        if InputValidator.check_sql_injection(text):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input detected"
            )

        # Check for XSS
        if InputValidator.check_xss(text):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input detected"
            )

        # Sanitize based on HTML allowance
        if allow_html:
            return InputValidator.sanitize_html(text, allowed_tags=['p', 'br', 'strong', 'em'])
        else:
            return InputValidator.escape_html(text)

    @staticmethod
    def validate_file_upload(
        filename: str,
        content_type: str,
        file_size: int,
        allowed_extensions: Optional[List[str]] = None,
        max_size: int = 10 * 1024 * 1024
    ) -> tuple[bool, str]:
        """
        Validate file upload.
        Returns (is_valid, error_message)
        """
        # Check file size
        if file_size > max_size:
            return False, f"File size exceeds maximum of {max_size / (1024*1024)}MB"

        # Check filename
        if not filename or '..' in filename or '/' in filename or '\\' in filename:
            return False, "Invalid filename"

        # Check extension
        if allowed_extensions:
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if ext not in allowed_extensions:
                return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"

        return True, ""

    @staticmethod
    def sanitize_dict(data: Dict[str, Any], max_depth: int = 10, current_depth: int = 0) -> Dict[str, Any]:
        """Recursively sanitize dictionary values."""
        if current_depth > max_depth:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data too deeply nested"
            )

        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = InputValidator.escape_html(value)
            elif isinstance(value, dict):
                sanitized[key] = InputValidator.sanitize_dict(value, max_depth, current_depth + 1)
            elif isinstance(value, list):
                sanitized[key] = [
                    InputValidator.escape_html(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                sanitized[key] = value

        return sanitized


# Create validator instance
validator = InputValidator()
