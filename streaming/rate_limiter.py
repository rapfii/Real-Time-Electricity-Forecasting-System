"""
Token bucket rate limiter for streaming simulation.

Allows controlled bursting (up to capacity tokens instantly) then
sustained throughput at refill_rate. Uses asyncio.sleep() for
non-blocking waiting — no time.sleep(), no spin-waiting.

Reference: SYSTEM_DESIGN.md Section 5.1
"""

import asyncio
import logging
import time

from config.settings import Settings

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Token bucket rate limiter with async acquire.

    Attributes:
        capacity: Maximum tokens in the bucket (burst allowance).
        refill_rate: Tokens added per second.
        tokens: Current token count.
        last_refill: Monotonic timestamp of last refill.
    """

    def __init__(
        self,
        capacity: int | None = None,
        refill_rate: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        s = settings or Settings()
        self.capacity: int = capacity or s.token_bucket_capacity
        self.refill_rate: float = refill_rate or s.token_bucket_refill_rate
        self.tokens: float = float(self.capacity)
        self.last_refill: float = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill = now

    async def acquire(self, n: int = 1) -> None:
        """Acquire n tokens, waiting asynchronously if necessary.

        Args:
            n: Number of tokens to consume (default 1).

        Algorithm:
            1. Compute elapsed time since last refill
            2. Add elapsed * refill_rate tokens (capped at capacity)
            3. If tokens >= n: consume and return
            4. Else: compute wait_time = (n - tokens) / refill_rate
            5. await asyncio.sleep(wait_time)
            6. Refill and consume
        """
        self._refill()

        if self.tokens >= n:
            self.tokens -= n
            return

        # Not enough tokens — compute wait time
        deficit = n - self.tokens
        wait_time = deficit / self.refill_rate

        # Non-blocking yield to event loop
        await asyncio.sleep(wait_time)

        # Refill after sleeping and consume
        self._refill()
        self.tokens -= n

    @property
    def tokens_available(self) -> float:
        """Return current available tokens (with refill calculation)."""
        self._refill()
        return self.tokens
