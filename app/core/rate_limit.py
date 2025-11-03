"""Rate limiting middleware using Redis"""

import redis.asyncio as redis
from fastapi import HTTPException, status, Request
from app.core.config import settings
import time


class RateLimiter:
    """Rate limiter using Redis for distributed rate limiting"""

    def __init__(self):
        self.redis_client = None

    async def init_redis(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
            password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
            encoding="utf-8",
            decode_responses=True
        )

    async def close_redis(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

    async def check_rate_limit(self, request: Request, api_key: str) -> bool:
        """
        Check if request is within rate limit

        Args:
            request: FastAPI request object
            api_key: API key from request

        Returns:
            True if within limit

        Raises:
            HTTPException: If rate limit exceeded
        """
        if not self.redis_client:
            await self.init_redis()

        # Create unique key for this API key
        key = f"rate_limit:{api_key}"
        current_time = int(time.time())
        window = 60  # 1 minute window

        # Remove old entries outside the window
        await self.redis_client.zremrangebyscore(key, 0, current_time - window)

        # Count requests in current window
        request_count = await self.redis_client.zcard(key)

        if request_count >= settings.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute."
            )

        # Add current request
        await self.redis_client.zadd(key, {str(current_time): current_time})
        await self.redis_client.expire(key, window)

        return True


# Global rate limiter instance
rate_limiter = RateLimiter()
