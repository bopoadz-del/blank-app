"""
Security configuration for production deployment.
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class SecuritySettings(BaseSettings):
    """Security settings for the application."""

    # CORS Settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]
    cors_max_age: int = 3600

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_calls: int = 100  # requests
    rate_limit_period: int = 60  # seconds
    rate_limit_storage: str = "memory"  # memory or redis

    # Request Size Limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_upload_size: int = 100 * 1024 * 1024  # 100MB

    # Session & Token Settings
    session_timeout: int = 3600  # 1 hour
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    token_algorithm: str = "HS256"

    # IP Filtering
    ip_whitelist: List[str] = []
    ip_blacklist: List[str] = []

    # Trusted Hosts
    allowed_hosts: List[str] = ["*"]

    # Security Headers
    enable_security_headers: bool = True
    enable_hsts: bool = True
    hsts_max_age: int = 31536000  # 1 year
    enable_csp: bool = True

    # SSL/TLS
    force_https: bool = False
    ssl_redirect: bool = False

    # API Keys
    api_key_header_name: str = "X-API-Key"
    api_key_rotation_days: int = 90

    # Audit Logging
    audit_log_enabled: bool = True
    audit_log_file: str = "logs/audit.log"
    audit_log_retention_days: int = 90

    # Input Validation
    enable_sql_injection_check: bool = True
    enable_xss_check: bool = True
    enable_path_traversal_check: bool = True

    # Password Policy
    min_password_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special_char: bool = True
    password_history: int = 5

    # Account Security
    max_login_attempts: int = 5
    lockout_duration: int = 1800  # 30 minutes
    enable_2fa: bool = False

    # Database Security
    db_ssl_required: bool = False
    db_connection_timeout: int = 30
    db_pool_size: int = 20
    db_max_overflow: int = 10

    # Encryption
    encryption_algorithm: str = "AES-256-GCM"
    encrypt_sensitive_data: bool = True

    # Environment
    environment: str = "production"
    debug_mode: bool = False

    class Config:
        env_file = ".env"
        env_prefix = "SECURITY_"


class ProductionSecuritySettings(SecuritySettings):
    """Production-specific security settings."""

    # Stricter CORS
    cors_origins: List[str] = []  # Must be explicitly set

    # Higher rate limits
    rate_limit_calls: int = 1000
    rate_limit_period: int = 60

    # Force HTTPS
    force_https: bool = True
    ssl_redirect: bool = True
    enable_hsts: bool = True

    # Trusted hosts only
    allowed_hosts: List[str] = []  # Must be explicitly set

    # Stronger token expiry
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 1

    # Enable all security checks
    enable_sql_injection_check: bool = True
    enable_xss_check: bool = True
    enable_path_traversal_check: bool = True

    # Stricter password policy
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special_char: bool = True

    # Enhanced account security
    max_login_attempts: int = 3
    lockout_duration: int = 3600  # 1 hour
    enable_2fa: bool = True

    # Database security
    db_ssl_required: bool = True

    environment: str = "production"
    debug_mode: bool = False


class DevelopmentSecuritySettings(SecuritySettings):
    """Development-specific security settings."""

    # Relaxed CORS for development
    cors_origins: List[str] = ["*"]

    # Lower rate limits for testing
    rate_limit_calls: int = 50
    rate_limit_period: int = 60

    # No HTTPS requirement
    force_https: bool = False
    ssl_redirect: bool = False

    # Allow all hosts in development
    allowed_hosts: List[str] = ["*"]

    # Relaxed password policy
    min_password_length: int = 6
    require_uppercase: bool = False
    require_lowercase: bool = False
    require_digit: bool = False
    require_special_char: bool = False

    # Relaxed account security
    max_login_attempts: int = 10
    lockout_duration: int = 300  # 5 minutes
    enable_2fa: bool = False

    environment: str = "development"
    debug_mode: bool = True


def get_security_settings() -> SecuritySettings:
    """Get security settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionSecuritySettings()
    elif env == "development":
        return DevelopmentSecuritySettings()
    else:
        return SecuritySettings()


# Export settings
security_settings = get_security_settings()
