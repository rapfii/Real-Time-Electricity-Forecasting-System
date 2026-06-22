"""
REST API endpoints.

Endpoints:
  POST /predict         - Single-shot inference
  GET  /metrics         - Current evaluation metrics
  GET  /health          - System health check
  GET  /stream/status   - Streaming simulation state

Reference: SYSTEM_DESIGN.md Section 14.1
"""

import logging
import time
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, HTTPException

from api.app import get_pipeline
from api.schemas import (
    HealthResponse,
    MetricsResponse,
    MetricValues,
    LatencyMetrics,
    PredictRequest,
    PredictResponse,
    RateLimiterStatus,
    StreamStatusResponse,
)
from config.settings import Settings
from core.pricing_engine import PricingEngine
from streaming.feature_buffer import FeatureBuffer

logger = logging.getLogger(__name__)

router = APIRouter()
_settings = Settings()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Single-shot inference for a given load sequence.

    Accepts 168 hourly values for all 6 regions and returns
    a load forecast and price forecast for the next hour.
    """
    pipeline = get_pipeline()

    # Build a temporary feature buffer from the request data
    buf = FeatureBuffer(settings=_settings)

    seq = request.load_sequence
    n = min(
        len(seq.pjme), len(seq.pjmw), len(seq.aep),
        len(seq.dayton), len(seq.dom), len(seq.duq),
    )

    if n < _settings.feature_buffer_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least {_settings.feature_buffer_size} hourly values, "
                f"got {n}."
            ),
        )

    # Fill buffer with the provided sequence
    base_ts = request.timestamp - timedelta(hours=n)
    for i in range(n):
        ts = base_ts + timedelta(hours=i + 1)
        buf.update(
            pjme_mw=seq.pjme[i],
            pjmw_mw=seq.pjmw[i],
            aep_mw=seq.aep[i],
            dayton_mw=seq.dayton[i],
            dom_mw=seq.dom[i],
            duq_mw=seq.duq[i],
            timestamp=ts,
        )

    # Extract features and predict
    features = buf.get_features()

    start_ns = time.perf_counter_ns()
    from models.forecaster import LoadForecaster

    forecaster = pipeline._forecaster
    if not forecaster.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    load_hat, latency_ns = forecaster.predict(features)

    pricing = PricingEngine(_settings)
    price_hat = pricing.calculate(load_hat)

    predicted_ts = request.timestamp + timedelta(hours=1)

    return PredictResponse(
        timestamp_predicted=predicted_ts,
        load_forecast_mw=round(load_hat, 1),
        price_forecast_usd_mwh=round(price_hat, 2),
        inference_latency_ms=round(latency_ns / 1_000_000, 2),
        model_version=forecaster.model_version,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Retrieve current evaluation metrics."""
    pipeline = get_pipeline()
    m = pipeline.metrics_collector.latest_metrics

    if not m:
        return MetricsResponse(
            load_metrics=MetricValues(mae=0.0, rmse=0.0, mape=0.0),
            price_metrics=MetricValues(mae=0.0, rmse=0.0, mape=0.0),
            latency=LatencyMetrics(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0),
            events_processed=0,
            uptime_seconds=0.0,
        )

    return MetricsResponse(
        load_metrics=MetricValues(**m["load_metrics"]),
        price_metrics=MetricValues(**m["price_metrics"]),
        latency=LatencyMetrics(**m["latency"]),
        events_processed=m["events_processed"],
        uptime_seconds=m["uptime_seconds"],
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health check."""
    try:
        pipeline = get_pipeline()
        return HealthResponse(
            status="healthy",
            model_loaded=pipeline._forecaster.is_loaded,
            stream_active=pipeline.is_running,
            buffer_warm=pipeline.feature_buffer.is_warm(),
            events_in_queue=pipeline._event_queue.qsize(),
        )
    except RuntimeError:
        return HealthResponse(
            status="initializing",
            model_loaded=False,
            stream_active=False,
            buffer_warm=False,
            events_in_queue=0,
        )


@router.get("/stream/status", response_model=StreamStatusResponse)
async def stream_status() -> StreamStatusResponse:
    """Current streaming simulation state."""
    pipeline = get_pipeline()
    status = pipeline.get_stream_status()

    return StreamStatusResponse(
        logical_time=status["logical_time"],
        wall_clock_elapsed_s=status["wall_clock_elapsed_s"],
        events_emitted=status["events_emitted"],
        events_remaining=status["events_remaining"],
        progress_pct=status["progress_pct"],
        rate_limiter=RateLimiterStatus(**status["rate_limiter"]),
    )
