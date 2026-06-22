"""
System-wide configuration for rt-electricity-forecast.

All configuration values are defined here and sourced from environment
variables prefixed with RTEF_. Defaults match SYSTEM_DESIGN.md Appendix B.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Centralized configuration via Pydantic BaseSettings.

    All modules import configuration from this class. No hardcoded
    values should exist in individual modules.
    """

    model_config = ConfigDict(env_file=".env", env_prefix="RTEF_")

    # ── Data ──────────────────────────────────────────────────────────
    data_path: str = "data/pjm_hourly_est.csv"
    train_ratio: float = 0.80
    datetime_col: str = "Datetime"
    target_col: str = "PJME"
    auxiliary_cols: list[str] = ["PJMW", "AEP", "DAYTON", "DOM", "DUQ"]
    all_region_cols: list[str] = ["PJME", "PJMW", "AEP", "DAYTON", "DOM", "DUQ"]

    # ── Features – PJME lags ─────────────────────────────────────────
    pjme_lag_hours: list[int] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        48, 168,
    ]
    # Features – Cross-regional lags (per auxiliary region)
    cross_regional_lag_hours: list[int] = [1, 24]
    # Features – Rolling windows
    pjme_rolling_windows: list[int] = [24, 48, 168]
    cross_regional_rolling_windows: list[int] = [24]
    feature_buffer_size: int = 168
    total_features: int = 65

    # ── Model ─────────────────────────────────────────────────────────
    model_type: str = "lightgbm"
    model_path: str = "artifacts/model.lgb"
    n_estimators: int = 2000
    learning_rate: float = 0.05
    num_leaves: int = 127
    early_stopping_rounds: int = 50
    walk_forward_folds: int = 5

    # ── Pricing ───────────────────────────────────────────────────────
    price_intercept: float = 15.0
    price_linear_coeff: float = 0.005
    price_quadratic_coeff: float = 0.000002
    price_noise_std: float = 2.0

    # ── Streaming ─────────────────────────────────────────────────────
    token_bucket_capacity: int = 10
    token_bucket_refill_rate: float = 5.0
    metrics_update_interval: int = 24
    queue_max_size: int = 100

    # ── API ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501
