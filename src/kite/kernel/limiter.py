"""Client-side rate limiting: a requests bucket and a tokens bucket, with an injectable clock."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from kite.kernel.base import Kernel, estimate_tokens
from kite.kernel.types import KernelRequest, KernelResponse


class RateLimiter:
    """Two token buckets. `rpm` bounds requests per minute, `tps` bounds input tokens per second."""

    def __init__(
        self,
        rpm: float,
        tps: float,
        *,
        burst_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._request_rate = rpm / 60.0
        self._token_rate = float(tps)
        self._request_capacity = max(1.0, self._request_rate * burst_seconds)
        self._token_capacity = max(1.0, self._token_rate * burst_seconds)
        self._requests = self._request_capacity
        self._tokens = self._token_capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._penalty_factor = 1.0
        self._penalty_until = 0.0
        self._lock = asyncio.Lock()

    def _refill(self) -> float:
        now = self._clock()
        factor = self._penalty_factor if now < self._penalty_until else 1.0
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._requests = min(self._request_capacity, self._requests + elapsed * self._request_rate * factor)
        self._tokens = min(self._token_capacity, self._tokens + elapsed * self._token_rate * factor)
        return factor

    async def acquire(self, estimated_tokens: int) -> None:
        """Block until one request and `estimated_tokens` tokens are available, then take them."""
        need = min(float(estimated_tokens), self._token_capacity)
        async with self._lock:
            while True:
                factor = self._refill()
                if self._requests >= 1.0 and self._tokens >= need:
                    self._requests -= 1.0
                    self._tokens -= need
                    return
                wait_requests = (1.0 - self._requests) / (self._request_rate * factor)
                wait_tokens = (need - self._tokens) / (self._token_rate * factor)
                await self._sleep(max(wait_requests, wait_tokens, 1e-3))

    def reconcile(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Correct the token bucket once the real usage is known."""
        charged = min(float(estimated_tokens), self._token_capacity)
        self._tokens = max(-self._token_capacity, self._tokens - (float(actual_tokens) - charged))

    def penalize(self, factor: float = 0.5, duration: float = 60.0) -> None:
        """Slow down after the server says we are too fast."""
        self._refill()
        self._penalty_factor = factor
        self._penalty_until = self._clock() + duration


def is_rate_limit_error(error: BaseException) -> bool:
    if "RateLimit" in type(error).__name__:
        return True
    return 429 in (getattr(error, "status", None), getattr(error, "status_code", None))


class LimitedKernel:
    def __init__(self, inner: Kernel, limiter: RateLimiter) -> None:
        self._inner = inner
        self._limiter = limiter
        self.name = inner.name
        self.model_id = inner.model_id

    async def evaluate(self, request: KernelRequest) -> KernelResponse:
        estimate = estimate_tokens(request)
        await self._limiter.acquire(estimate)
        try:
            response = await self._inner.evaluate(request)
        except Exception as error:
            if is_rate_limit_error(error):
                self._limiter.penalize()
            raise
        self._limiter.reconcile(estimate, response.usage.input_tokens)
        return response
