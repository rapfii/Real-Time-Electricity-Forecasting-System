"""
Data loading, validation, and chronological splitting.

Reads pjm_hourly_est.csv, filters to 6 used regions, validates data
quality, and splits chronologically into training and streaming sets.

Reference: SYSTEM_DESIGN.md Sections 4.1, 6.1, 6.1.1, 6.2, 6.3
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


class DataLoader:
    """CSV ingestion, validation, and chronological split.

    Loads the consolidated PJM hourly dataset, filters to 6 target regions,
    drops rows with any null in those columns, and provides a chronological
    train/stream split.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

    def load(self) -> pd.DataFrame:
        """Load and validate the PJM hourly dataset.

        Returns:
            DataFrame with Datetime index and 6 region columns, sorted
            chronologically with all rows having complete data.
        """
        data_path = Path(self._settings.data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                f"Expected: {self._settings.data_path}"
            )

        logger.info("Loading dataset from %s", data_path)
        df = pd.read_csv(data_path)

        # Select only used columns: Datetime + 6 regions
        keep_cols = [self._settings.datetime_col] + self._settings.all_region_cols
        missing_cols = [c for c in keep_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df[keep_cols].copy()

        # Parse datetime
        df[self._settings.datetime_col] = pd.to_datetime(
            df[self._settings.datetime_col]
        )

        # Sort chronologically
        df = df.sort_values(self._settings.datetime_col).reset_index(drop=True)

        # Drop duplicate timestamps (keep last)
        n_before = len(df)
        df = df.drop_duplicates(
            subset=[self._settings.datetime_col], keep="last"
        ).reset_index(drop=True)
        n_dupes = n_before - len(df)
        if n_dupes > 0:
            logger.warning("Dropped %d duplicate timestamps.", n_dupes)

        # Drop rows where any of the 6 regions is null
        n_before = len(df)
        df = df.dropna(subset=self._settings.all_region_cols).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                "Dropped %d rows with null values in region columns. "
                "%d complete rows remain.",
                n_dropped,
                len(df),
            )

        # Validate: clip negative load values to 0.0 with warning
        for col in self._settings.all_region_cols:
            neg_mask = df[col] < 0
            if neg_mask.any():
                n_neg = neg_mask.sum()
                logger.warning(
                    "Clipped %d negative values in %s to 0.0.", n_neg, col
                )
                df.loc[neg_mask, col] = 0.0

        # Flag extreme outliers (> mu + 5*sigma) — do NOT remove
        for col in self._settings.all_region_cols:
            mu = df[col].mean()
            sigma = df[col].std()
            outlier_mask = df[col] > mu + 5 * sigma
            if outlier_mask.any():
                logger.warning(
                    "Column %s has %d values exceeding μ+5σ (%.0f). "
                    "These are flagged but retained (energy spikes may be real).",
                    col,
                    outlier_mask.sum(),
                    mu + 5 * sigma,
                )

        # Check for missing hours (gaps in the time series)
        time_diffs = df[self._settings.datetime_col].diff().dt.total_seconds() / 3600
        gaps = time_diffs[time_diffs > 1.0]
        if len(gaps) > 0:
            logger.warning(
                "Found %d time gaps exceeding 1 hour in the dataset.",
                len(gaps),
            )
            large_gaps = gaps[gaps > 3.0]
            if len(large_gaps) > 0:
                logger.warning(
                    "%d gaps exceed 3 hours. These are flagged for review.",
                    len(large_gaps),
                )

        logger.info(
            "Dataset loaded: %d rows, time range %s to %s.",
            len(df),
            df[self._settings.datetime_col].iloc[0],
            df[self._settings.datetime_col].iloc[-1],
        )

        return df

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset chronologically into train and stream sets.

        Args:
            df: Full validated dataset from load().

        Returns:
            Tuple of (train_df, stream_df). Train is first 80%, stream
            is the remaining 20%. No shuffling.
        """
        n = len(df)
        split_idx = int(n * self._settings.train_ratio)

        train_df = df.iloc[:split_idx].reset_index(drop=True)
        stream_df = df.iloc[split_idx:].reset_index(drop=True)

        logger.info(
            "Chronological split: %d training rows, %d streaming rows "
            "(ratio=%.2f).",
            len(train_df),
            len(stream_df),
            self._settings.train_ratio,
        )

        return train_df, stream_df
