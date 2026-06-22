"""
Baseline models for comparison against LightGBM.

These are NOT production models. They exist only to validate that the
primary LightGBM model outperforms simple heuristics.

Reference: SYSTEM_DESIGN.md Section 9.3

Baselines:
  1. Naive persistence:   Load_hat(t+1) = Load(t)
  2. Seasonal naive:      Load_hat(t+1) = Load(t-168)  (same hour last week)
  3. 24h moving average:  Load_hat(t+1) = mean(Load(t), ..., Load(t-23))
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


class NaivePersistence:
    """Baseline: predict next-hour load as current load.

    Load_hat(t+1) = Load(t)
    """

    def predict(self, load_series: pd.Series) -> pd.Series:
        """Generate naive persistence predictions.

        Args:
            load_series: Chronological PJME load values.

        Returns:
            Series of predictions (shifted by 1). First value will be NaN.
        """
        return load_series.shift(1)


class SeasonalNaive:
    """Baseline: predict next-hour load as same hour last week.

    Load_hat(t+1) = Load(t-168)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or Settings()
        self._seasonal_lag: int = s.feature_buffer_size  # 168

    def predict(self, load_series: pd.Series) -> pd.Series:
        """Generate seasonal naive predictions.

        Args:
            load_series: Chronological PJME load values.

        Returns:
            Series of predictions (shifted by 168). First 168 values are NaN.
        """
        return load_series.shift(self._seasonal_lag)


class MovingAverage24h:
    """Baseline: predict next-hour load as 24h moving average.

    Load_hat(t+1) = mean(Load(t), Load(t-1), ..., Load(t-23))
    """

    def predict(self, load_series: pd.Series) -> pd.Series:
        """Generate 24-hour moving average predictions.

        Args:
            load_series: Chronological PJME load values.

        Returns:
            Series of predictions. First 23 values will be NaN.
        """
        return load_series.rolling(window=24, min_periods=24).mean()


def evaluate_baselines(
    actual: pd.Series,
    load_series: pd.Series,
) -> dict[str, dict[str, float]]:
    """Run all baselines and compute metrics.

    Args:
        actual: Actual next-hour load values (ground truth).
        load_series: Current-hour load values for generating predictions.

    Returns:
        Dictionary mapping baseline name to its metrics (MAE, RMSE, MAPE).
    """
    from core.metrics import mae, rmse, mape

    baselines = {
        "naive_persistence": NaivePersistence(),
        "seasonal_naive": SeasonalNaive(),
        "moving_average_24h": MovingAverage24h(),
    }

    results: dict[str, dict[str, float]] = {}
    for name, model in baselines.items():
        preds = model.predict(load_series)
        # Align: drop NaN rows from both actual and predictions
        mask = preds.notna() & actual.notna()
        a = actual[mask].values
        p = preds[mask].values

        results[name] = {
            "mae": mae(a, p),
            "rmse": rmse(a, p),
            "mape": mape(a, p),
            "n_samples": int(mask.sum()),
        }
        logger.info(
            "Baseline %s: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
            name,
            results[name]["mae"],
            results[name]["rmse"],
            results[name]["mape"],
        )

    return results
