"""
CLI entrypoint for offline model training.

Runs the full training pipeline:
1. Load configuration
2. Load and validate dataset
3. Split chronologically
4. Run walk-forward training
5. Save model and training report

Reference: SYSTEM_DESIGN.md Section 17.3 steps 1-4
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from core.data_loader import DataLoader
from models.trainer import WalkForwardTrainer
from models.baselines import evaluate_baselines


def main() -> None:
    """Run offline training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("RT-ELECTRICITY-FORECAST: Training Pipeline")
    logger.info("=" * 60)

    settings = Settings()

    # Step 1: Load and validate dataset
    logger.info("Step 1: Loading dataset...")
    loader = DataLoader(settings)
    df = loader.load()

    # Step 2: Split chronologically
    logger.info("Step 2: Splitting dataset...")
    train_df, stream_df = loader.split(df)
    logger.info(
        "Training: %d rows, Streaming: %d rows",
        len(train_df),
        len(stream_df),
    )

    # Step 3: Run walk-forward training
    logger.info("Step 3: Running walk-forward training...")
    trainer = WalkForwardTrainer(settings)
    report = trainer.train(train_df)

    # Step 4: Evaluate baselines for comparison
    logger.info("Step 4: Evaluating baselines...")
    # Use the feature-engineered data for baseline comparison
    from core.feature_engine import FeatureEngineer

    fe = FeatureEngineer(settings)
    feature_df = fe.build_features(train_df)
    target = feature_df["target"]
    current_load = feature_df["pjme_lag_1"]  # proxy for current load

    baseline_results = evaluate_baselines(target, current_load)

    # Add baselines to report
    report["baseline_results"] = baseline_results

    # Update training report
    report_path = Path(settings.model_path).parent / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("-" * 60)
    logger.info(
        "Model: %s (%d features, %d training rows)",
        settings.model_type,
        report["n_features"],
        report["n_training_rows"],
    )
    logger.info(
        "Avg Validation: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        report["average_metrics"]["mae"],
        report["average_metrics"]["rmse"],
        report["average_metrics"]["mape"],
    )
    logger.info("Model saved to: %s", report["model_path"])
    logger.info("Report saved to: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
