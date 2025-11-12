"""Application configuration"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Formula Execution API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Server Configuration
    HOST: str = "0.0.0.0"  # nosec B104 - Intentional for container deployments
    PORT: int = 8000

    # Security
    SECRET_KEY: str = "dev-secret-key-CHANGE-IN-PRODUCTION"  # Must be set via environment variable in production
    API_KEY: str = "test-api-key-12345-CHANGE-IN-PRODUCTION"  # Must be set via environment variable in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/formulas"  # Must use environment variable in production
    DATABASE_ECHO: bool = False

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 10
    RATE_LIMIT_BURST: int = 5

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
