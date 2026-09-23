import asyncio

import pytest

from kite.kernel.limiter import LimitedKernel, RateLimiter, is_rate_limit_error
from kite.kernel.mock import MockKernel
from kite.kernel.types import KernelRequest, QuestionSpec


class FakeClock:
    """Time only moves when the limiter sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_limiter(clock: FakeClock, rpm: float, tps: float) -> RateLimiter:
    return RateLimiter(rpm, tps, burst_seconds=2.0, clock=clock, sleep=clock.sleep)


def test_request_bucket_allows_a_burst_then_paces():
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=60, tps=1e9)  # 1 request/s, burst capacity 2

    async def five():
        for _ in range(5):
            await limiter.acquire(1)

    asyncio.run(five())
    assert clock.now == pytest.approx(3.0, abs=0.05)


def test_token_bucket_paces_large_requests():
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=1e9, tps=100)  # capacity 200 tokens

    async def three():
        for _ in range(3):
            await limiter.acquire(150)

    asyncio.run(three())
    assert clock.now == pytest.approx(2.5, abs=0.05)


def test_oversized_request_is_clamped_instead_of_deadlocking():
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=1e9, tps=100)
    asyncio.run(limiter.acquire(10_000))
    assert clock.now == 0.0


def test_penalty_halves_the_rate():
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=60, tps=1e9)
    limiter.penalize(factor=0.5, duration=60.0)

    async def four():
        for _ in range(4):
            await limiter.acquire(1)

    asyncio.run(four())
    assert clock.now == pytest.approx(4.0, abs=0.05)


def test_reconcile_charges_the_difference():
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=1e9, tps=100)

    async def scenario():
        await limiter.acquire(50)  # 150 left
        limiter.reconcile(50, 150)  # really used 150 -> 50 left
        await limiter.acquire(100)  # needs 50 more -> waits 0.5 s

    asyncio.run(scenario())
    assert clock.now == pytest.approx(0.5, abs=0.05)


def test_is_rate_limit_error():
    class TypeSafeRateLimitError(Exception):
        pass

    class WithStatus(Exception):
        status_code = 429

    assert is_rate_limit_error(TypeSafeRateLimitError())
    assert is_rate_limit_error(WithStatus())
    assert not is_rate_limit_error(ValueError())


def test_limited_kernel_penalizes_on_rate_limit_and_reraises():
    class TypeSafeRateLimitError(Exception):
        pass

    class Throttled:
        name, model_id = "jev", "jev-1.13.0"

        async def evaluate(self, request):
            raise TypeSafeRateLimitError()

    clock = FakeClock()
    limiter = make_limiter(clock, rpm=60, tps=1e9)
    request = KernelRequest(state="s", questions={"q": QuestionSpec(type="noul", instructions="x")})
    with pytest.raises(TypeSafeRateLimitError):
        asyncio.run(LimitedKernel(Throttled(), limiter).evaluate(request))
    assert limiter._penalty_until == pytest.approx(60.0)


def test_limited_kernel_passes_responses_through():
    clock = FakeClock()
    kernel = LimitedKernel(MockKernel(), make_limiter(clock, rpm=60, tps=1e9))
    request = KernelRequest(state="s", questions={"q": QuestionSpec(type="noul", instructions="x")})
    response = asyncio.run(kernel.evaluate(request))
    assert response.backend == "mock"
    assert (kernel.name, kernel.model_id) == ("mock", "mock-1")
