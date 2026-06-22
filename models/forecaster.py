"""
Online inference wrapper for the trained LightGBM model.

Loads a serialized model and provides single-sample prediction with
latency measurement. Clips negative predictions to 0.

Reference: SYSTEM_DESIGN.md Sections 10.1, 10.3
"""

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from config.settings import Settings

logger = logging.getLogger(__name__)


class LoadForecaster:
    """Online load forecaster using a serialized LightGBM model.

    Loads the trained model from disk and provides predict() for
    single-sample inference with latency tracking.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._model: lgb.Booster | None = None
        self._model_version: str = ""

    def load_model(self, model_path: str | None = None) -> None:
        """Load a serialized LightGBM model from disk.

        Args:
            model_path: Path to the .lgb model file. Uses settings default
                        if not provided.
        """
        path = Path(model_path or self._settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        self._model = lgb.Booster(model_file=str(path))
        self._model_version = path.stem
        logger.info("Model loaded from %s (version: %s)", path, self._model_version)

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None

    @property
    def model_version(self) -> str:
        """Return the current model version identifier."""
        return self._model_version

    def predict(self, features: np.ndarray) -> tuple[float, int]:
        """Run single-sample inference.

        Args:
            features: Feature array of shape (1, 65).

        Returns:
            Tuple of (predicted_load_mw, latency_ns).
            Negative predictions are clipped to 0.0.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start_ns = time.perf_counter_ns()
        prediction = self._model.predict(features)
        latency_ns = time.perf_counter_ns() - start_ns

        load_hat = float(prediction[0])

        # Clip negative predictions (physically impossible)
        if load_hat < 0.0:
            logger.warning(
                "Negative load prediction (%.2f MW) clipped to 0.0.", load_hat
            )
            load_hat = 0.0

        return load_hat, latency_ns
