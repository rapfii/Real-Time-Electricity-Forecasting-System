"""
Ring buffer for online feature construction during streaming.

Maintains deques of the last 168 values for PJME and each auxiliary
region, and constructs the 65-feature vector on demand. The buffer
must be warm (168 values) before predictions are emitted.

Reference: SYSTEM_DESIGN.md Section 10.2
"""

import logging
from collections import deque
from datetime import datetime

import numpy as np

from config.settings import Settings

logger = logging.getLogger(__name__)


class FeatureBuffer:
    """Ring buffer maintaining online feature state for streaming inference.

    Stores the last 168 multi-region load values and timestamps.
    Constructs the full 65-feature vector from the buffer state.

    Attributes:
        pjme_buffer: Deque of last 168 PJME load values.
        region_buffers: Dict of deques for each auxiliary region.
        timestamps: Deque of last 168 timestamps.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        buf_size = self._settings.feature_buffer_size  # 168

        self.pjme_buffer: deque[float] = deque(maxlen=buf_size)
        self.region_buffers: dict[str, deque[float]] = {
            region.lower(): deque(maxlen=buf_size)
            for region in self._settings.auxiliary_cols
        }
        self.timestamps: deque[datetime] = deque(maxlen=buf_size)

    def update(
        self,
        pjme_mw: float,
        pjmw_mw: float,
        aep_mw: float,
        dayton_mw: float,
        dom_mw: float,
        duq_mw: float,
        timestamp: datetime,
    ) -> None:
        """Append new values to all ring buffers.

        Args:
            pjme_mw: PJME load value in MW.
            pjmw_mw: PJMW load value in MW.
            aep_mw: AEP load value in MW.
            dayton_mw: DAYTON load value in MW.
            dom_mw: DOM load value in MW.
            duq_mw: DUQ load value in MW.
            timestamp: Event timestamp.
        """
        self.pjme_buffer.append(pjme_mw)
        self.region_buffers["pjmw"].append(pjmw_mw)
        self.region_buffers["aep"].append(aep_mw)
        self.region_buffers["dayton"].append(dayton_mw)
        self.region_buffers["dom"].append(dom_mw)
        self.region_buffers["duq"].append(duq_mw)
        self.timestamps.append(timestamp)

    def is_warm(self) -> bool:
        """Check if buffer has enough data for feature construction.

        Returns:
            True if the buffer contains at least 168 values.
        """
        return len(self.pjme_buffer) >= self._settings.feature_buffer_size

    def get_features(self) -> np.ndarray:
        """Extract all 65 features from current buffer state.

        Returns:
            Feature array of shape (1, 65) for single-sample prediction.

        Raises:
            RuntimeError: If buffer is not warm (< 168 values).
        """
        if not self.is_warm():
            raise RuntimeError(
                f"Buffer not warm: {len(self.pjme_buffer)} values, "
                f"need {self._settings.feature_buffer_size}."
            )

        pjme = list(self.pjme_buffer)  # index 0 = oldest, -1 = newest
        current_ts = self.timestamps[-1]
        features: list[float] = []

        # ── Category 1: PJME lag features (26) ──────────────────────
        # pjme[-1] is current value (t), pjme[-2] is t-1, etc.
        for lag in self._settings.pjme_lag_hours:
            idx = -lag  # offset from end
            features.append(pjme[idx])

        # ── Category 2: Cross-regional lag features (10) ────────────
        for region in ["pjmw", "aep", "dayton", "dom", "duq"]:
            buf = list(self.region_buffers[region])
            for lag in self._settings.cross_regional_lag_hours:
                idx = -lag
                features.append(buf[idx])

        # ── Category 3: PJME rolling statistics (8) ─────────────────
        pjme_arr = np.array(pjme)

        for window in self._settings.pjme_rolling_windows:
            # Use values from t-1 back (shift by 1 to avoid lookahead)
            window_data = pjme_arr[-(window + 1):-1]
            features.append(float(np.mean(window_data)))   # mean
            features.append(float(np.std(window_data, ddof=1) if len(window_data) > 1 else 0.0))  # std

        # min/max for 24h window
        window_24h = pjme_arr[-25:-1]  # 24 values from t-1 to t-24
        features.append(float(np.min(window_24h)))    # min_24h
        features.append(float(np.max(window_24h)))    # max_24h

        # ── Category 4: Cross-regional rolling features (5) ─────────
        for region in ["pjmw", "aep", "dayton", "dom", "duq"]:
            buf_arr = np.array(list(self.region_buffers[region]))
            for window in self._settings.cross_regional_rolling_windows:
                window_data = buf_arr[-(window + 1):-1]
                features.append(float(np.mean(window_data)))

        # ── Category 5: Temporal features (11) ──────────────────────
        hour = current_ts.hour
        dow = current_ts.weekday()
        month = current_ts.month
        day_of_year = current_ts.timetuple().tm_yday

        features.append(float(hour))
        features.append(float(np.sin(2 * np.pi * hour / 24)))
        features.append(float(np.cos(2 * np.pi * hour / 24)))
        features.append(float(dow))
        features.append(float(np.sin(2 * np.pi * dow / 7)))
        features.append(float(np.cos(2 * np.pi * dow / 7)))
        features.append(float(month))
        features.append(float(np.sin(2 * np.pi * month / 12)))
        features.append(float(np.cos(2 * np.pi * month / 12)))
        features.append(float(1 if dow >= 5 else 0))
        features.append(float(day_of_year))

        # ── Category 6: PJME derived features (3) ───────────────────
        pjme_current = pjme[-1]
        pjme_1h_ago = pjme[-2]
        pjme_24h_ago = pjme[-25]

        features.append(pjme_current - pjme_1h_ago)       # diff_1h
        features.append(pjme_current - pjme_24h_ago)      # diff_24h
        # Avoid division by zero
        ratio_24h = pjme_current / pjme_24h_ago if pjme_24h_ago != 0 else 1.0
        features.append(ratio_24h)                          # ratio_24h

        # ── Category 7: Cross-regional ratio features (2) ───────────
        pjmw_current = list(self.region_buffers["pjmw"])[-1]
        aep_current = list(self.region_buffers["aep"])[-1]
        dayton_current = list(self.region_buffers["dayton"])[-1]
        dom_current = list(self.region_buffers["dom"])[-1]
        duq_current = list(self.region_buffers["duq"])[-1]

        # pjme_to_pjmw_ratio
        pjme_to_pjmw = pjme_current / pjmw_current if pjmw_current != 0 else 1.0
        features.append(pjme_to_pjmw)

        # pjme_to_total_ratio
        total_load = (
            pjme_current + pjmw_current + aep_current
            + dayton_current + dom_current + duq_current
        )
        pjme_to_total = pjme_current / total_load if total_load != 0 else 0.0
        features.append(pjme_to_total)

        # Verify feature count
        assert len(features) == self._settings.total_features, (
            f"Expected {self._settings.total_features} features, "
            f"got {len(features)}"
        )

        return np.array(features, dtype=np.float64).reshape(1, -1)
