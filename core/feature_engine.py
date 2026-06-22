"""
Batch feature engineering for training.

Transforms raw multi-region load DataFrame into a feature matrix with
exactly 65 features in 7 categories. This module is stateless and used
for batch (offline) feature construction during training.

Reference: SYSTEM_DESIGN.md Sections 7.1 through 7.8

Feature Vector (65 total):
  Category 1 – PJME lag features (26)
  Category 2 – Cross-regional lag features (10)
  Category 3 – PJME rolling statistics (8)
  Category 4 – Cross-regional rolling features (5)
  Category 5 – Temporal features (11)
  Category 6 – PJME derived features (3)
  Category 7 – Cross-regional ratio features (2)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Stateless batch feature transformation (training mode).

    Processes a chronologically sorted DataFrame of multi-region load
    data and produces a feature matrix with exactly 65 columns.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the complete 65-feature matrix from raw load data.

        Args:
            df: DataFrame with Datetime column and 6 region load columns
                (PJME, PJMW, AEP, DAYTON, DOM, DUQ), sorted chronologically.

        Returns:
            DataFrame with 65 feature columns plus the target column.
            Rows with NaN features (due to insufficient lag history) are dropped.
        """
        logger.info("Building features from %d rows...", len(df))
        features = pd.DataFrame(index=df.index)

        # ── Category 1: PJME lag features (26) ──────────────────────
        target = self._settings.target_col
        for lag in self._settings.pjme_lag_hours:
            col_name = f"pjme_lag_{lag}"
            features[col_name] = df[target].shift(lag)

        # ── Category 2: Cross-regional lag features (10) ────────────
        for region in self._settings.auxiliary_cols:
            r = region.lower()
            for lag in self._settings.cross_regional_lag_hours:
                col_name = f"{r}_lag_{lag}"
                features[col_name] = df[region].shift(lag)

        # ── Category 3: PJME rolling statistics (8) ─────────────────
        for window in self._settings.pjme_rolling_windows:
            features[f"pjme_roll_mean_{window}h"] = (
                df[target].shift(1).rolling(window=window, min_periods=window).mean()
            )
            features[f"pjme_roll_std_{window}h"] = (
                df[target].shift(1).rolling(window=window, min_periods=window).std()
            )

        # min/max only for 24h window
        features["pjme_roll_min_24h"] = (
            df[target].shift(1).rolling(window=24, min_periods=24).min()
        )
        features["pjme_roll_max_24h"] = (
            df[target].shift(1).rolling(window=24, min_periods=24).max()
        )

        # ── Category 4: Cross-regional rolling features (5) ─────────
        for region in self._settings.auxiliary_cols:
            r = region.lower()
            for window in self._settings.cross_regional_rolling_windows:
                col_name = f"{r}_roll_mean_{window}h"
                features[col_name] = (
                    df[region]
                    .shift(1)
                    .rolling(window=window, min_periods=window)
                    .mean()
                )

        # ── Category 5: Temporal features (11) ──────────────────────
        dt = df[self._settings.datetime_col]
        features["hour_of_day"] = dt.dt.hour
        features["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        features["day_of_week"] = dt.dt.dayofweek
        features["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        features["month"] = dt.dt.month
        features["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        features["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        features["day_of_year"] = dt.dt.dayofyear

        # ── Category 6: PJME derived features (3) ───────────────────
        features["pjme_diff_1h"] = df[target] - df[target].shift(1)
        features["pjme_diff_24h"] = df[target] - df[target].shift(24)
        features["pjme_ratio_24h"] = df[target] / df[target].shift(24)

        # ── Category 7: Cross-regional ratio features (2) ───────────
        features["pjme_to_pjmw_ratio"] = df["PJME"] / df["PJMW"]
        total_load = sum(df[col] for col in self._settings.all_region_cols)
        features["pjme_to_total_ratio"] = df["PJME"] / total_load

        # Add target: next-hour PJME load
        features["target"] = df[target].shift(-1)

        # Drop rows with any NaN features (due to lag/rolling warmup)
        n_before = len(features)
        features = features.dropna().reset_index(drop=True)
        n_dropped = n_before - len(features)
        logger.info(
            "Dropped %d rows with NaN features. %d rows remain.",
            n_dropped,
            len(features),
        )

        # Verify feature count
        feature_cols = [c for c in features.columns if c != "target"]
        n_features = len(feature_cols)
        if n_features != self._settings.total_features:
            raise ValueError(
                f"Expected {self._settings.total_features} features, "
                f"got {n_features}. Features: {feature_cols}"
            )

        logger.info(
            "Feature matrix built: %d rows × %d features.",
            len(features),
            n_features,
        )

        return features

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of all 65 feature names.

        Returns:
            List of feature column names in canonical order.
        """
        names: list[str] = []

        # Category 1: PJME lags (26)
        for lag in self._settings.pjme_lag_hours:
            names.append(f"pjme_lag_{lag}")

        # Category 2: Cross-regional lags (10)
        for region in self._settings.auxiliary_cols:
            r = region.lower()
            for lag in self._settings.cross_regional_lag_hours:
                names.append(f"{r}_lag_{lag}")

        # Category 3: PJME rolling stats (8)
        for window in self._settings.pjme_rolling_windows:
            names.append(f"pjme_roll_mean_{window}h")
            names.append(f"pjme_roll_std_{window}h")
        names.append("pjme_roll_min_24h")
        names.append("pjme_roll_max_24h")

        # Category 4: Cross-regional rolling (5)
        for region in self._settings.auxiliary_cols:
            r = region.lower()
            for window in self._settings.cross_regional_rolling_windows:
                names.append(f"{r}_roll_mean_{window}h")

        # Category 5: Temporal (11)
        names.extend([
            "hour_of_day", "hour_sin", "hour_cos",
            "day_of_week", "dow_sin", "dow_cos",
            "month", "month_sin", "month_cos",
            "is_weekend", "day_of_year",
        ])

        # Category 6: PJME derived (3)
        names.extend(["pjme_diff_1h", "pjme_diff_24h", "pjme_ratio_24h"])

        # Category 7: Cross-regional ratios (2)
        names.extend(["pjme_to_pjmw_ratio", "pjme_to_total_ratio"])

        assert len(names) == self._settings.total_features, (
            f"Expected {self._settings.total_features} features, got {len(names)}"
        )

        return names
