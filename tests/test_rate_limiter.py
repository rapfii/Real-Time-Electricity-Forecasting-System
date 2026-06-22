"""Unit tests for the TokenBucketRateLimiter.

Tests:
  - Initial state has full capacity
  - Acquire consumes tokens
  - Refill adds tokens over time
  - Burst behavior (up to capacity)
  - Async wait when bucket is empty
"""

import asyncio
import time

import pytest

from streaming.rate_limiter import TokenBucketRateLimiter


@pytest.fixture
def limiter() -> TokenBucketRateLimiter:
    """Create a rate limiter with capacity=5, refill_rate=10."""
    return TokenBucketRateLimiter(capacity=5, refill_rate=10.0)


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_initial_tokens(self, limiter: TokenBucketRateLimiter) -> None:
        """Bucket starts full at capacity."""
        assert limiter.tokens == 5.0

    @pytest.mark.asyncio
    async def test_acquire_consumes_token(
        self, limiter: TokenBucketRateLimiter
    ) -> None:
        """Acquiring should reduce token count."""
        await limiter.acquire(1)
        assert limiter.tokens < 5.0

    @pytest.mark.asyncio
    async def test_burst_up_to_capacity(
        self, limiter: TokenBucketRateLimiter
    ) -> None:
        """Can acquire up to capacity instantly."""
        for _ in range(5):
            await limiter.acquire(1)
        # All 5 tokens consumed (approximately, allowing for refill during test)
        assert limiter.tokens_available < 1.0

    @pytest.mark.asyncio
    async def test_refill_over_time(self) -> None:
        """Tokens should refill based on elapsed time."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=100.0)
        # Consume all tokens
        for _ in range(10):
            await limiter.acquire(1)

        # Wait for refill
        await asyncio.sleep(0.05)  # 50ms at 100/s = ~5 tokens
        available = limiter.tokens_available
        assert available > 0

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self) -> None:
        """Acquiring when empty should wait (not raise)."""
        limiter = TokenBucketRateLimiter(capacity=1, refill_rate=100.0)
        await limiter.acquire(1)
        # This should wait briefly and then succeed
        start = time.monotonic()
        await limiter.acquire(1)
        elapsed = time.monotonic() - start
        # Should have waited but not too long
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_no_exceed_capacity(self) -> None:
        """Tokens should never exceed capacity after refill."""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=100.0)
        await asyncio.sleep(0.1)  # Wait for lots of potential refill
        available = limiter.tokens_available
        assert available <= 5.0

    @pytest.mark.asyncio
    async def test_acquire_multiple(self) -> None:
        """Can acquire more than 1 token at a time."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=5.0)
        await limiter.acquire(5)
        assert limiter.tokens_available < 6.0
