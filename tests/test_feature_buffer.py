"""Unit tests for the FeatureBuffer.

Tests:
  - Buffer starts cold (not warm)
  - Becomes warm after 168 updates
  - get_features() returns shape (1, 65)
  - Raises when not warm
  - Ring buffer behavior (maxlen)
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from config.settings import Settings
from streaming.feature_buffer import FeatureBuffer


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def buffer(settings: Settings) -> FeatureBuffer:
    return FeatureBuffer(settings)


def _fill_buffer(buf: FeatureBuffer, n: int) -> None:
    """Fill buffer with n synthetic events."""
    base_time = datetime(2017, 1, 1, 0, 0, 0)
    for i in range(n):
        buf.update(
            pjme_mw=30000.0 + i * 10,
            pjmw_mw=8000.0 + i * 5,
            aep_mw=15000.0 + i * 7,
            dayton_mw=2500.0 + i * 2,
            dom_mw=12000.0 + i * 6,
            duq_mw=1800.0 + i * 1,
            timestamp=base_time + timedelta(hours=i),
        )


class TestFeatureBuffer:
    """Tests for FeatureBuffer."""

    def test_initial_not_warm(self, buffer: FeatureBuffer) -> None:
        """Buffer starts cold."""
        assert not buffer.is_warm()

    def test_warm_after_168(self, buffer: FeatureBuffer) -> None:
        """Buffer becomes warm after 168 updates."""
        _fill_buffer(buffer, 168)
        assert buffer.is_warm()

    def test_not_warm_at_167(self, buffer: FeatureBuffer) -> None:
        """Buffer is not warm at 167 updates."""
        _fill_buffer(buffer, 167)
        assert not buffer.is_warm()

    def test_get_features_raises_when_cold(
        self, buffer: FeatureBuffer
    ) -> None:
        """get_features() raises RuntimeError when buffer is cold."""
        _fill_buffer(buffer, 100)
        with pytest.raises(RuntimeError, match="Buffer not warm"):
            buffer.get_features()

    def test_get_features_shape(self, buffer: FeatureBuffer) -> None:
        """get_features() returns shape (1, 65)."""
        _fill_buffer(buffer, 168)
        features = buffer.get_features()
        assert features.shape == (1, 65)

    def test_get_features_no_nan(self, buffer: FeatureBuffer) -> None:
        """Feature vector should contain no NaN values."""
        _fill_buffer(buffer, 168)
        features = buffer.get_features()
        assert not np.isnan(features).any()

    def test_ring_buffer_maxlen(self, buffer: FeatureBuffer) -> None:
        """Buffer respects maxlen (168) — old values are evicted."""
        _fill_buffer(buffer, 200)
        assert len(buffer.pjme_buffer) == 168
        assert buffer.is_warm()

    def test_feature_dtype(self, buffer: FeatureBuffer) -> None:
        """Features should be float64."""
        _fill_buffer(buffer, 168)
        features = buffer.get_features()
        assert features.dtype == np.float64

    def test_multi_region_storage(self, buffer: FeatureBuffer) -> None:
        """Buffer stores data for all 5 auxiliary regions."""
        _fill_buffer(buffer, 168)
        assert len(buffer.region_buffers) == 5
        for region in ["pjmw", "aep", "dayton", "dom", "duq"]:
            assert region in buffer.region_buffers
            assert len(buffer.region_buffers[region]) == 168

    def test_timestamps_stored(self, buffer: FeatureBuffer) -> None:
        """Buffer stores timestamps."""
        _fill_buffer(buffer, 168)
        assert len(buffer.timestamps) == 168
