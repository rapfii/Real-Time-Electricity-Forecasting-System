"""
Pydantic request/response schemas for the API.

Reference: SYSTEM_DESIGN.md Sections 14.1, 14.2
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── POST /predict ────────────────────────────────────────────────────

class LoadSequenceInput(BaseModel):
    """Multi-region load sequence for prediction."""

    pjme: list[float] = Field(..., description="PJME load values (168 hourly values)")
    pjmw: list[float] = Field(..., description="PJMW load values (168 hourly values)")
    aep: list[float] = Field(..., description="AEP load values (168 hourly values)")
    dayton: list[float] = Field(..., description="DAYTON load values (168 hourly values)")
    dom: list[float] = Field(..., description="DOM load values (168 hourly values)")
    duq: list[float] = Field(..., description="DUQ load values (168 hourly values)")


class PredictRequest(BaseModel):
    """Request body for POST /predict endpoint."""

    load_sequence: LoadSequenceInput
    timestamp: datetime


class PredictResponse(BaseModel):
    """Response body for POST /predict endpoint."""

    timestamp_predicted: datetime
    load_forecast_mw: float
    price_forecast_usd_mwh: float
    inference_latency_ms: float
    model_version: str


# ── GET /metrics ─────────────────────────────────────────────────────

class MetricValues(BaseModel):
    """Load or price metric values."""

    mae: float
    rmse: float
    mape: float


class LatencyMetrics(BaseModel):
    """Latency percentile metrics."""

    p50_ms: float
    p95_ms: float
    p99_ms: float


class MetricsResponse(BaseModel):
    """Response body for GET /metrics endpoint."""

    load_metrics: MetricValues
    price_metrics: MetricValues
    latency: LatencyMetrics
    events_processed: int
    uptime_seconds: float


# ── GET /health ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health endpoint."""

    status: str
    model_loaded: bool
    stream_active: bool
    buffer_warm: bool
    events_in_queue: int


# ── GET /stream/status ───────────────────────────────────────────────

class RateLimiterStatus(BaseModel):
    """Token bucket rate limiter status."""

    tokens_available: float
    capacity: int
    refill_rate: float


class StreamStatusResponse(BaseModel):
    """Response body for GET /stream/status endpoint."""

    logical_time: str | None
    wall_clock_elapsed_s: float
    events_emitted: int
    events_remaining: int
    progress_pct: float
    rate_limiter: RateLimiterStatus


# ── WebSocket Messages ───────────────────────────────────────────────

class ForecastMessage(BaseModel):
    """WebSocket per-event forecast message."""

    type: str = "forecast"
    sequence_id: int
    logical_timestamp: str
    load_actual_mw: float
    load_forecast_mw: float
    load_error_mw: float
    price_actual_usd_mwh: float
    price_forecast_usd_mwh: float
    price_error_usd_mwh: float
    inference_latency_ms: float


class MetricsUpdateMessage(BaseModel):
    """WebSocket periodic metrics update message."""

    type: str = "metrics_update"
    window: str = "24h_rolling"
    load_mae: float
    load_rmse: float
    price_mae: float
    price_rmse: float
    latency_p99_ms: float
