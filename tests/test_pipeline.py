"""Integration tests for the streaming pipeline.

Tests:
  - Replay engine emits correct LoadEvent structure
  - Pipeline processes events through two-stage inference
  - GroundTruthSimulator adds noise
  - MetricsCollector accumulates results
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from streaming.pipeline import (
    ForecastResult,
    GroundTruthSimulator,
    MetricsCollector,
)
from streaming.replay_engine import LoadEvent, LogicalTimeReplayEngine
from streaming.rate_limiter import TokenBucketRateLimiter


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def sample_stream_df() -> pd.DataFrame:
    """Create a small synthetic stream DataFrame."""
    n = 200
    rng = np.random.default_rng(42)
    dates = pd.date_range("2017-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Datetime": dates,
        "PJME": rng.normal(30000, 3000, n).clip(10000),
        "PJMW": rng.normal(8000, 1000, n).clip(3000),
        "AEP": rng.normal(15000, 2000, n).clip(5000),
        "DAYTON": rng.normal(2500, 400, n).clip(1000),
        "DOM": rng.normal(12000, 1500, n).clip(4000),
        "DUQ": rng.normal(1800, 250, n).clip(500),
    })


class TestLoadEvent:
    """Tests for LoadEvent dataclass."""

    def test_all_fields(self) -> None:
        """LoadEvent carries all 6 region values."""
        event = LoadEvent(
            timestamp=datetime(2017, 1, 1),
            pjme_mw=30000.0,
            pjmw_mw=8000.0,
            aep_mw=15000.0,
            dayton_mw=2500.0,
            dom_mw=12000.0,
            duq_mw=1800.0,
            sequence_id=0,
        )
        assert event.pjme_mw == 30000.0
        assert event.pjmw_mw == 8000.0
        assert event.aep_mw == 15000.0
        assert event.dayton_mw == 2500.0
        assert event.dom_mw == 12000.0
        assert event.duq_mw == 1800.0
        assert event.sequence_id == 0


class TestReplayEngine:
    """Tests for LogicalTimeReplayEngine."""

    @pytest.mark.asyncio
    async def test_stream_emits_events(
        self, sample_stream_df: pd.DataFrame, settings: Settings
    ) -> None:
        """Replay engine should emit all events from DataFrame."""
        limiter = TokenBucketRateLimiter(capacity=1000, refill_rate=1000.0)
        engine = LogicalTimeReplayEngine(
            stream_df=sample_stream_df,
            rate_limiter=limiter,
            settings=settings,
        )

        events = []
        async for event in engine.stream():
            events.append(event)

        assert len(events) == 200

    @pytest.mark.asyncio
    async def test_stream_chronological_order(
        self, sample_stream_df: pd.DataFrame, settings: Settings
    ) -> None:
        """Events should be in chronological order."""
        limiter = TokenBucketRateLimiter(capacity=1000, refill_rate=1000.0)
        engine = LogicalTimeReplayEngine(
            stream_df=sample_stream_df,
            rate_limiter=limiter,
            settings=settings,
        )

        timestamps = []
        async for event in engine.stream():
            timestamps.append(event.timestamp)

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


class TestGroundTruthSimulator:
    """Tests for GroundTruthSimulator."""

    def test_adds_noise(self, settings: Settings) -> None:
        """Ground truth prices should include noise (not exactly match base)."""
        sim = GroundTruthSimulator(settings)
        prices = [sim.compute(30000.0) for _ in range(100)]
        # Prices should vary due to noise
        assert np.std(prices) > 0

    def test_noise_centered(self, settings: Settings) -> None:
        """Noise should be centered around the deterministic price."""
        from core.pricing_engine import PricingEngine

        sim = GroundTruthSimulator(settings)
        pe = PricingEngine(settings)
        base = pe.calculate(30000.0)

        prices = [sim.compute(30000.0) for _ in range(10000)]
        mean_price = np.mean(prices)
        # Mean should be close to base price (within noise tolerance)
        assert abs(mean_price - base) < 1.0


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_update_accumulates(self, settings: Settings) -> None:
        """MetricsCollector should accumulate results."""
        collector = MetricsCollector(settings)
        result = ForecastResult(
            timestamp=datetime(2017, 1, 1),
            load_forecast_mw=30500.0,
            price_forecast_usd_mwh=200.0,
            load_actual_mw=30000.0,
            price_actual_usd_mwh=198.0,
            latency_ns=500000,
            sequence_id=0,
        )
        metrics = collector.update(result)
        # First update (not at interval boundary) returns None
        assert metrics is None

    def test_metrics_at_interval(self, settings: Settings) -> None:
        """MetricsCollector returns metrics at update interval."""
        settings.metrics_update_interval = 5
        collector = MetricsCollector(settings)

        for i in range(5):
            result = ForecastResult(
                timestamp=datetime(2017, 1, 1) + timedelta(hours=i),
                load_forecast_mw=30000.0 + i * 100,
                price_forecast_usd_mwh=200.0 + i,
                load_actual_mw=30000.0,
                price_actual_usd_mwh=198.0,
                latency_ns=500000,
                sequence_id=i,
            )
            metrics = collector.update(result)

        # 5th update should trigger metrics
        assert metrics is not None
        assert "load_metrics" in metrics
        assert "price_metrics" in metrics
        assert "latency" in metrics
