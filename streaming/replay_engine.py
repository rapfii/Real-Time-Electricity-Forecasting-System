"""
Logical time replay engine for streaming simulation.

Replays historical load data as an async event stream governed by
a token bucket rate limiter. Logical time advances discretely by
1 hour per event, regardless of wall-clock time.

Reference: SYSTEM_DESIGN.md Section 5.2
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator

import pandas as pd

from config.settings import Settings
from streaming.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


@dataclass
class LoadEvent:
    """Single load event emitted by the replay engine.

    Carries all 6 region values plus metadata.
    Reference: SYSTEM_DESIGN.md Section 4.3, constraint B11.
    """

    timestamp: datetime
    pjme_mw: float
    pjmw_mw: float
    aep_mw: float
    dayton_mw: float
    dom_mw: float
    duq_mw: float
    sequence_id: int


class LogicalTimeReplayEngine:
    """Async generator yielding LoadEvents at a governed rate.

    Replays a chronologically sorted buffer of historical data through
    a token bucket rate limiter. Logical time advances by the inter-event
    interval in the dataset (1 hour), regardless of wall-clock time.

    Attributes:
        replay_buffer: List of LoadEvent objects, chronologically sorted.
        logical_clock: Current simulated timestamp.
        rate_limiter: TokenBucketRateLimiter governing emission rate.
    """

    def __init__(
        self,
        stream_df: pd.DataFrame,
        rate_limiter: TokenBucketRateLimiter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(
            settings=self._settings
        )
        self.logical_clock: datetime | None = None

        # Build replay buffer from DataFrame
        self.replay_buffer: list[LoadEvent] = []
        for idx, row in stream_df.iterrows():
            event = LoadEvent(
                timestamp=row[self._settings.datetime_col].to_pydatetime()
                if isinstance(row[self._settings.datetime_col], pd.Timestamp)
                else row[self._settings.datetime_col],
                pjme_mw=float(row["PJME"]),
                pjmw_mw=float(row["PJMW"]),
                aep_mw=float(row["AEP"]),
                dayton_mw=float(row["DAYTON"]),
                dom_mw=float(row["DOM"]),
                duq_mw=float(row["DUQ"]),
                sequence_id=int(idx),
            )
            self.replay_buffer.append(event)

        self.total_events: int = len(self.replay_buffer)
        logger.info(
            "Replay engine initialized with %d events.", self.total_events
        )

    async def stream(self) -> AsyncGenerator[LoadEvent, None]:
        """Async generator that yields LoadEvents at governed rate.

        For each event:
            1. await rate_limiter.acquire(1)
            2. Update logical_clock
            3. Yield event
        """
        for event in self.replay_buffer:
            await self.rate_limiter.acquire(1)
            self.logical_clock = event.timestamp
            yield event
