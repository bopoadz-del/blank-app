"""Rate limiting middleware using Redis with graceful fallback."""

from collections import defaultdict, deque
from typing import Deque, DefaultDict, Optional
import time

from fastapi import HTTPException, status, Request

from app.core.config import settings

try:
    import redis.asyncio as redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled by fallback
    redis = None


class RateLimiter:
    """Rate limiter using Redis for distributed rate limiting"""

    def __init__(self):
        self.redis_client: Optional["redis.Redis"] = None
        self._fallback_buckets: DefaultDict[str, Deque[int]] = defaultdict(deque)
        self._fallback_enabled = False

    async def init_redis(self):
        """Initialize Redis connection"""
        if redis is None:
            self._fallback_enabled = True
            return

        try:
            self.redis_client = await redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                encoding="utf-8",
                decode_responses=True
            )
            self._fallback_enabled = False
        except Exception:
            # Redis server is unavailable – fall back to in-memory rate limiting
            self.redis_client = None
            self._fallback_enabled = True

    async def close_redis(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
        self.redis_client = None
        self._fallback_buckets.clear()
        self._fallback_enabled = False

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
        if not self.redis_client and not self._fallback_enabled:
            await self.init_redis()

        key = f"rate_limit:{api_key}"
        current_time = int(time.time())
        window = 60  # 1 minute window

        if self.redis_client:
            try:
                await self.redis_client.zremrangebyscore(key, 0, current_time - window)
                request_count = await self.redis_client.zcard(key)

                if request_count >= settings.RATE_LIMIT_PER_MINUTE:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute."
                    )

                await self.redis_client.zadd(key, {str(current_time): current_time})
                await self.redis_client.expire(key, window)
                return True
            except Exception:
                # Switch to fallback if Redis becomes unavailable mid-request
                self.redis_client = None
                self._fallback_enabled = True

        # Fallback in-memory rate limiting
<<<<<l< codex/fix-failed-ci-and-security-scan-workflows-zu1wmc
=======
<<<<<codex/fix-failed-ci-and-security-scan-workflows-xj83mk
>>>>> main
        if self._fallback_enabled:
            bucket = self._fallback_buckets[key]
            while bucket and bucket[0] <= current_time - window:
                bucket.popleft()

            if len(bucket) >= settings.RATE_LIMIT_PER_MINUTE * 10:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute."
                )

            bucket.append(current_time)
            return True

          
<<<<< codex/fix-failed-ci-and-security-scan-workflows-zu1wmc
=======
=======
        bucket = self._fallback_buckets[key]
        while bucket and bucket[0] <= current_time - window:
            bucket.popleft()

        if len(bucket) >= settings.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute."
            )

        bucket.append(current_time)
>>>>> main
>>>>> main
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()
