"""
CLI entrypoint for batch evaluation.

Runs the trained model on the streaming replay buffer in batch mode
(without the async pipeline) and generates a comprehensive evaluation
report comparing against baselines.

Reference: SYSTEM_DESIGN.md Section 17.2 (make evaluate)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from core.data_loader import DataLoader
from core.feature_engine import FeatureEngineer
from core.metrics import mae, rmse, mape, latency_percentiles
from core.pricing_engine import PricingEngine
from models.forecaster import LoadForecaster
from models.baselines import evaluate_baselines


def main() -> None:
    """Run batch evaluation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("RT-ELECTRICITY-FORECAST: Batch Evaluation")
    logger.info("=" * 60)

    settings = Settings()

    # Load dataset
    loader = DataLoader(settings)
    df = loader.load()
    _, stream_df = loader.split(df)
    logger.info("Evaluation set: %d rows", len(stream_df))

    # Build features for the streaming set
    fe = FeatureEngineer(settings)
    feature_df = fe.build_features(stream_df)
    feature_names = fe.get_feature_names()

    X = feature_df[feature_names].values
    y = feature_df["target"].values
    logger.info("Feature matrix: %d rows × %d features", len(X), len(feature_names))

    # Load trained model
    forecaster = LoadForecaster(settings)
    forecaster.load_model()

    # Run predictions
    pricing = PricingEngine(settings)
    rng = np.random.default_rng(seed=42)

    load_predictions: list[float] = []
    price_predictions: list[float] = []
    price_actuals: list[float] = []
    latencies_ns: list[int] = []

    for i in range(len(X)):
        features = X[i:i + 1]
        load_hat, latency_ns = forecaster.predict(features)
        price_hat = pricing.calculate(load_hat)

        load_actual = y[i]
        price_base = pricing.calculate(load_actual)
        noise = rng.normal(0.0, settings.price_noise_std)
        price_actual = price_base + noise

        load_predictions.append(load_hat)
        price_predictions.append(price_hat)
        price_actuals.append(price_actual)
        latencies_ns.append(latency_ns)

    load_actuals = y.tolist()

    # Compute metrics
    load_mae_val = mae(load_actuals, load_predictions)
    load_rmse_val = rmse(load_actuals, load_predictions)
    load_mape_val = mape(load_actuals, load_predictions)

    price_mae_val = mae(price_actuals, price_predictions)
    price_rmse_val = rmse(price_actuals, price_predictions)
    price_mape_val = mape(price_actuals, price_predictions)

    latency = latency_percentiles(latencies_ns)

    # Baseline comparison
    logger.info("Running baseline evaluation...")
    current_load = feature_df["pjme_lag_1"]
    target_series = feature_df["target"]
    baseline_results = evaluate_baselines(target_series, current_load)

    # Build report
    report = {
        "evaluation_set_rows": len(X),
        "model_metrics": {
            "load": {
                "mae": load_mae_val,
                "rmse": load_rmse_val,
                "mape": load_mape_val,
            },
            "price": {
                "mae": price_mae_val,
                "rmse": price_rmse_val,
                "mape": price_mape_val,
            },
        },
        "latency": latency,
        "baseline_results": baseline_results,
        "model_beats_all_baselines": all(
            load_mae_val < b["mae"] for b in baseline_results.values()
        ),
    }

    # Save report
    report_path = Path("artifacts/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("-" * 60)
    logger.info(
        "LightGBM: Load MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        load_mae_val,
        load_rmse_val,
        load_mape_val,
    )
    logger.info(
        "LightGBM: Price MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        price_mae_val,
        price_rmse_val,
        price_mape_val,
    )
    for name, bm in baseline_results.items():
        logger.info(
            "Baseline %s: Load MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
            name,
            bm["mae"],
            bm["rmse"],
            bm["mape"],
        )
    logger.info("Latency: p50=%.2fms, p95=%.2fms, p99=%.2fms",
                latency["p50_ms"], latency["p95_ms"], latency["p99_ms"])
    logger.info(
        "Model beats all baselines: %s",
        report["model_beats_all_baselines"],
    )
    logger.info("Report saved to: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
