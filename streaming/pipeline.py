"""
Streaming orchestration pipeline.

Wires together: replay engine → feature buffer → forecaster → pricing
engine → metrics collector. Uses asyncio.Queue for backpressure-aware
communication between components. No threading in the hot path.

Reference: SYSTEM_DESIGN.md Sections 5.3, 4.4, 4.5
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings
from core.metrics import mae, rmse, mape, latency_percentiles
from core.pricing_engine import PricingEngine
from models.forecaster import LoadForecaster
from streaming.feature_buffer import FeatureBuffer
from streaming.replay_engine import LoadEvent, LogicalTimeReplayEngine

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result of a single two-stage inference.

    Reference: SYSTEM_DESIGN.md Section 10.1
    """

    timestamp: datetime
    load_forecast_mw: float
    price_forecast_usd_mwh: float
    load_actual_mw: float
    price_actual_usd_mwh: float
    latency_ns: int
    sequence_id: int


class MetricsCollector:
    """Accumulates predictions and actuals, computes rolling metrics.

    Every N events (configurable, default 24), computes load metrics,
    price metrics, and latency percentiles.

    Reference: SYSTEM_DESIGN.md Section 4.5
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._update_interval: int = self._settings.metrics_update_interval

        self._load_actuals: list[float] = []
        self._load_predictions: list[float] = []
        self._price_actuals: list[float] = []
        self._price_predictions: list[float] = []
        self._latencies_ns: list[int] = []
        self._events_processed: int = 0
        self._start_time: float = time.monotonic()

        # Latest computed metrics
        self.latest_metrics: dict[str, Any] = {}

    def update(self, result: ForecastResult) -> dict[str, Any] | None:
        """Record a forecast result and optionally compute metrics.

        Args:
            result: A ForecastResult from the inference pipeline.

        Returns:
            Metrics dictionary if update interval reached, else None.
        """
        self._load_actuals.append(result.load_actual_mw)
        self._load_predictions.append(result.load_forecast_mw)
        self._price_actuals.append(result.price_actual_usd_mwh)
        self._price_predictions.append(result.price_forecast_usd_mwh)
        self._latencies_ns.append(result.latency_ns)
        self._events_processed += 1

        if self._events_processed % self._update_interval == 0:
            metrics = self._compute_metrics()
            self.latest_metrics = metrics
            return metrics

        return None

    def _compute_metrics(self) -> dict[str, Any]:
        """Compute current evaluation metrics."""
        latency = latency_percentiles(self._latencies_ns)
        elapsed = time.monotonic() - self._start_time

        metrics = {
            "load_metrics": {
                "mae": mae(self._load_actuals, self._load_predictions),
                "rmse": rmse(self._load_actuals, self._load_predictions),
                "mape": mape(self._load_actuals, self._load_predictions),
            },
            "price_metrics": {
                "mae": mae(self._price_actuals, self._price_predictions),
                "rmse": rmse(self._price_actuals, self._price_predictions),
                "mape": mape(self._price_actuals, self._price_predictions),
            },
            "latency": latency,
            "events_processed": self._events_processed,
            "uptime_seconds": round(elapsed, 1),
        }

        return metrics

    def get_final_metrics(self) -> dict[str, Any]:
        """Compute and return final metrics."""
        return self._compute_metrics()


class GroundTruthSimulator:
    """Computes ground-truth price from actual load with Gaussian noise.

    Price_actual = PricingEngine.calculate(Load_actual) + ε
    where ε ~ N(0, σ²) with σ = price_noise_std (default 2.0).

    The noise is added HERE, not inside the PricingEngine.
    Reference: SYSTEM_DESIGN.md Section 4.4, B2
    """

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or Settings()
        self._pricing_engine = PricingEngine(s)
        self._noise_std: float = s.price_noise_std
        self._rng = np.random.default_rng(seed=42)

    def compute(self, load_actual_mw: float) -> float:
        """Compute ground-truth price with noise.

        Args:
            load_actual_mw: Actual load in megawatts.

        Returns:
            Simulated actual price in $/MWh.
        """
        base_price = self._pricing_engine.calculate(load_actual_mw)
        noise = self._rng.normal(0.0, self._noise_std)
        return base_price + noise


class StreamingPipeline:
    """Orchestrates the full streaming inference pipeline.

    Topology (from SYSTEM_DESIGN.md Section 5.3):
        replay_engine.stream()       → produces LoadEvents
        inference_pipeline.process() → consumes events, produces forecasts
        metrics_collector.update()   → consumes forecasts + actuals
        websocket_publisher.push()   → publishes to dashboard

    All components communicate via asyncio.Queue with bounded capacity.
    No threading in the hot path.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._feature_buffer = FeatureBuffer(self._settings)
        self._pricing_engine = PricingEngine(self._settings)
        self._ground_truth = GroundTruthSimulator(self._settings)
        self._metrics_collector = MetricsCollector(self._settings)
        self._forecaster = LoadForecaster(self._settings)

        # Queues for inter-component communication
        self._event_queue: asyncio.Queue[LoadEvent | None] = asyncio.Queue(
            maxsize=self._settings.queue_max_size
        )
        self._result_queue: asyncio.Queue[ForecastResult | None] = asyncio.Queue(
            maxsize=self._settings.queue_max_size
        )

        # State tracking
        self._events_emitted: int = 0
        self._events_total: int = 0
        self._is_running: bool = False
        self._warmup_count: int = 0
        self._logical_time: datetime | None = None
        self._wall_clock_start: float = 0.0

        # WebSocket broadcast callback (set by API layer)
        self._ws_broadcast: Any = None

    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self._is_running

    @property
    def metrics_collector(self) -> MetricsCollector:
        """Access the metrics collector."""
        return self._metrics_collector

    @property
    def feature_buffer(self) -> FeatureBuffer:
        """Access the feature buffer."""
        return self._feature_buffer

    @property
    def logical_time(self) -> datetime | None:
        """Current logical simulation time."""
        return self._logical_time

    @property
    def events_emitted(self) -> int:
        """Number of events emitted so far."""
        return self._events_emitted

    @property
    def events_total(self) -> int:
        """Total events in replay buffer."""
        return self._events_total

    def set_ws_broadcast(self, callback: Any) -> None:
        """Set WebSocket broadcast callback for live updates."""
        self._ws_broadcast = callback

    async def run(self, replay_engine: LogicalTimeReplayEngine) -> dict[str, Any]:
        """Run the full streaming pipeline to completion.

        Args:
            replay_engine: Initialized replay engine with stream data.

        Returns:
            Final evaluation metrics dictionary.
        """
        self._forecaster.load_model()
        self._is_running = True
        self._events_total = replay_engine.total_events
        self._wall_clock_start = time.monotonic()

        logger.info(
            "Starting streaming pipeline: %d events to process.",
            self._events_total,
        )

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(self._produce(replay_engine))
        consumer_task = asyncio.create_task(self._consume())

        await producer_task
        # Signal consumer to stop
        await self._event_queue.put(None)
        await consumer_task

        self._is_running = False
        final_metrics = self._metrics_collector.get_final_metrics()

        logger.info(
            "Pipeline complete. %d events processed. "
            "Load MAE=%.2f, Price MAE=%.2f",
            final_metrics["events_processed"],
            final_metrics["load_metrics"]["mae"],
            final_metrics["price_metrics"]["mae"],
        )

        return final_metrics

    async def _produce(self, replay_engine: LogicalTimeReplayEngine) -> None:
        """Producer: emit LoadEvents from replay engine into queue."""
        async for event in replay_engine.stream():
            await self._event_queue.put(event)
            self._events_emitted += 1
            self._logical_time = event.timestamp

    async def _consume(self) -> None:
        """Consumer: process LoadEvents from queue through inference pipeline."""
        while True:
            event = await self._event_queue.get()
            if event is None:
                break

            result = await self._process_event(event)
            if result is not None:
                # Update metrics
                metrics_update = self._metrics_collector.update(result)

                # Broadcast via WebSocket if callback set
                if self._ws_broadcast is not None:
                    forecast_msg = {
                        "type": "forecast",
                        "sequence_id": result.sequence_id,
                        "logical_timestamp": result.timestamp.isoformat(),
                        "load_actual_mw": result.load_actual_mw,
                        "load_forecast_mw": result.load_forecast_mw,
                        "load_error_mw": result.load_forecast_mw - result.load_actual_mw,
                        "price_actual_usd_mwh": result.price_actual_usd_mwh,
                        "price_forecast_usd_mwh": result.price_forecast_usd_mwh,
                        "price_error_usd_mwh": (
                            result.price_forecast_usd_mwh
                            - result.price_actual_usd_mwh
                        ),
                        "inference_latency_ms": result.latency_ns / 1_000_000,
                    }
                    try:
                        await self._ws_broadcast(json.dumps(forecast_msg))
                    except Exception:
                        pass

                    if metrics_update is not None:
                        metrics_msg = {
                            "type": "metrics_update",
                            "window": "24h_rolling",
                            "load_mae": metrics_update["load_metrics"]["mae"],
                            "load_rmse": metrics_update["load_metrics"]["rmse"],
                            "price_mae": metrics_update["price_metrics"]["mae"],
                            "price_rmse": metrics_update["price_metrics"]["rmse"],
                            "latency_p99_ms": metrics_update["latency"]["p99_ms"],
                        }
                        try:
                            await self._ws_broadcast(json.dumps(metrics_msg))
                        except Exception:
                            pass

    async def _process_event(self, event: LoadEvent) -> ForecastResult | None:
        """Process a single LoadEvent through the two-stage pipeline.

        Returns None during warm-up phase (first 168 events).
        """
        # Update feature buffer with all 6 region values
        self._feature_buffer.update(
            pjme_mw=event.pjme_mw,
            pjmw_mw=event.pjmw_mw,
            aep_mw=event.aep_mw,
            dayton_mw=event.dayton_mw,
            dom_mw=event.dom_mw,
            duq_mw=event.duq_mw,
            timestamp=event.timestamp,
        )

        # Warm-up phase: buffer not full yet
        if not self._feature_buffer.is_warm():
            self._warmup_count += 1
            return None

        # Stage 1: Load forecasting (ML)
        features = self._feature_buffer.get_features()  # shape (1, 65)
        load_hat, latency_ns = self._forecaster.predict(features)

        # Stage 2: Pricing engine (deterministic business logic)
        price_hat = self._pricing_engine.calculate(load_hat)

        # Ground truth
        load_actual = event.pjme_mw
        price_actual = self._ground_truth.compute(load_actual)

        return ForecastResult(
            timestamp=event.timestamp,
            load_forecast_mw=load_hat,
            price_forecast_usd_mwh=price_hat,
            load_actual_mw=load_actual,
            price_actual_usd_mwh=price_actual,
            latency_ns=latency_ns,
            sequence_id=event.sequence_id,
        )

    def get_stream_status(self) -> dict[str, Any]:
        """Get current streaming simulation state.

        Returns:
            Status dictionary matching SYSTEM_DESIGN.md Section 14.1
            GET /stream/status response.
        """
        elapsed = time.monotonic() - self._wall_clock_start if self._wall_clock_start else 0.0
        remaining = self._events_total - self._events_emitted
        progress = (
            (self._events_emitted / self._events_total * 100)
            if self._events_total > 0
            else 0.0
        )

        return {
            "logical_time": self._logical_time.isoformat() if self._logical_time else None,
            "wall_clock_elapsed_s": round(elapsed, 1),
            "events_emitted": self._events_emitted,
            "events_remaining": remaining,
            "progress_pct": round(progress, 1),
            "rate_limiter": {
                "tokens_available": 0.0,  # populated at runtime
                "capacity": self._settings.token_bucket_capacity,
                "refill_rate": self._settings.token_bucket_refill_rate,
            },
        }
