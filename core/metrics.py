"""
Evaluation metrics for load and price forecasting.

Pure functions: MAE, RMSE, MAPE. No side effects, no state.

Reference: SYSTEM_DESIGN.md Sections 16.1, 16.2, 16.3
"""

import math
import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Compute Mean Absolute Error.

    Formula: (1/n) * Σ|actual_i - predicted_i|

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        Mean absolute error.
    """
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if len(a) == 0:
        return 0.0
    return float(np.mean(np.abs(a - p)))


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Compute Root Mean Squared Error.

    Formula: sqrt((1/n) * Σ(actual_i - predicted_i)²)

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        Root mean squared error.
    """
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if len(a) == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Compute Mean Absolute Percentage Error.

    Formula: (100/n) * Σ|actual_i - predicted_i| / actual_i

    Args:
        actual: Ground truth values (must be non-zero).
        predicted: Predicted values.

    Returns:
        Mean absolute percentage error (as a percentage, e.g. 1.24 means 1.24%).
    """
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if len(a) == 0:
        return 0.0
    # Avoid division by zero
    mask = a != 0
    if not np.any(mask):
        logger.warning("All actual values are zero; MAPE is undefined.")
        return float("inf")
    return float(100.0 * np.mean(np.abs((a[mask] - p[mask]) / a[mask])))


def latency_percentiles(
    latencies_ns: Sequence[int],
) -> dict[str, float]:
    """Compute p50, p95, p99 latency percentiles.

    Args:
        latencies_ns: Sequence of latency measurements in nanoseconds.

    Returns:
        Dictionary with p50_ms, p95_ms, p99_ms keys (values in milliseconds).
    """
    if len(latencies_ns) == 0:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.asarray(latencies_ns, dtype=np.float64)
    # Convert nanoseconds to milliseconds
    arr_ms = arr / 1_000_000.0
    return {
        "p50_ms": float(np.percentile(arr_ms, 50)),
        "p95_ms": float(np.percentile(arr_ms, 95)),
        "p99_ms": float(np.percentile(arr_ms, 99)),
    }
