"""Unit tests for the FeatureEngineer.

Tests:
  - Produces exactly 65 features
  - Feature names match specification
  - Feature categories have correct counts
  - No NaN values in output
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from core.feature_engine import FeatureEngineer


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def feature_engineer(settings: Settings) -> FeatureEngineer:
    return FeatureEngineer(settings)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with 500 rows of synthetic data."""
    n = 500
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01-01", periods=n, freq="h")
    data = {
        "Datetime": dates,
        "PJME": rng.normal(30000, 5000, n).clip(10000),
        "PJMW": rng.normal(8000, 1500, n).clip(3000),
        "AEP": rng.normal(15000, 2500, n).clip(5000),
        "DAYTON": rng.normal(2500, 500, n).clip(1000),
        "DOM": rng.normal(12000, 2000, n).clip(4000),
        "DUQ": rng.normal(1800, 300, n).clip(500),
    }
    return pd.DataFrame(data)


class TestFeatureEngineer:
    """Tests for FeatureEngineer.build_features()."""

    def test_feature_count(
        self, feature_engineer: FeatureEngineer, sample_df: pd.DataFrame
    ) -> None:
        """Must produce exactly 65 features."""
        result = feature_engineer.build_features(sample_df)
        feature_cols = [c for c in result.columns if c != "target"]
        assert len(feature_cols) == 65

    def test_feature_names_match(
        self, feature_engineer: FeatureEngineer, sample_df: pd.DataFrame
    ) -> None:
        """Feature names must match get_feature_names()."""
        result = feature_engineer.build_features(sample_df)
        expected = feature_engineer.get_feature_names()
        actual = [c for c in result.columns if c != "target"]
        assert actual == expected

    def test_get_feature_names_count(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """get_feature_names() must return exactly 65 names."""
        names = feature_engineer.get_feature_names()
        assert len(names) == 65

    def test_pjme_lag_features(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """Category 1: 26 PJME lag features."""
        names = feature_engineer.get_feature_names()
        lag_features = [n for n in names if n.startswith("pjme_lag_")]
        assert len(lag_features) == 26

    def test_cross_regional_lag_features(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """Category 2: 10 cross-regional lag features."""
        names = feature_engineer.get_feature_names()
        regions = ["pjmw", "aep", "dayton", "dom", "duq"]
        cross_lags = [
            n
            for n in names
            if any(n.startswith(f"{r}_lag_") for r in regions)
        ]
        assert len(cross_lags) == 10

    def test_pjme_rolling_features(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """Category 3: 8 PJME rolling features."""
        names = feature_engineer.get_feature_names()
        rolling = [n for n in names if n.startswith("pjme_roll_")]
        assert len(rolling) == 8

    def test_cross_regional_rolling_features(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """Category 4: 5 cross-regional rolling features."""
        names = feature_engineer.get_feature_names()
        regions = ["pjmw", "aep", "dayton", "dom", "duq"]
        cross_roll = [
            n
            for n in names
            if any(n.startswith(f"{r}_roll_") for r in regions)
        ]
        assert len(cross_roll) == 5

    def test_temporal_features(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """Category 5: 11 temporal features."""
        names = feature_engineer.get_feature_names()
        temporal = [
            "hour_of_day", "hour_sin", "hour_cos",
            "day_of_week", "dow_sin", "dow_cos",
            "month", "month_sin", "month_cos",
            "is_weekend", "day_of_year",
        ]
        for t in temporal:
            assert t in names
        assert len(temporal) == 11

    def test_derived_features(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """Category 6: 3 PJME derived features."""
        names = feature_engineer.get_feature_names()
        derived = ["pjme_diff_1h", "pjme_diff_24h", "pjme_ratio_24h"]
        for d in derived:
            assert d in names

    def test_ratio_features(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """Category 7: 2 cross-regional ratio features."""
        names = feature_engineer.get_feature_names()
        ratios = ["pjme_to_pjmw_ratio", "pjme_to_total_ratio"]
        for r in ratios:
            assert r in names

    def test_no_nan_in_output(
        self, feature_engineer: FeatureEngineer, sample_df: pd.DataFrame
    ) -> None:
        """Output should have no NaN values after warmup rows are dropped."""
        result = feature_engineer.build_features(sample_df)
        assert not result.isnull().any().any()

    def test_has_target_column(
        self, feature_engineer: FeatureEngineer, sample_df: pd.DataFrame
    ) -> None:
        """Output must include a 'target' column."""
        result = feature_engineer.build_features(sample_df)
        assert "target" in result.columns
