"""
Walk-forward training pipeline with LightGBM.

Implements expanding-window walk-forward validation across 5 chronological
folds. No random splits. No sklearn KFold/TimeSeriesSplit.

Reference: SYSTEM_DESIGN.md Sections 9.1, 9.2, 8.2

Walk-forward folds (on 92,950 training rows):
  Fold 1: Train [0:18590]   → Validate [18590:37180]
  Fold 2: Train [0:37180]   → Validate [37180:55770]
  Fold 3: Train [0:55770]   → Validate [55770:74360]
  Fold 4: Train [0:74360]   → Validate [74360:92950]
  Fold 5: Train [0:92950]   → Final production model (full corpus)
"""

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.settings import Settings
from core.feature_engine import FeatureEngineer
from core.metrics import mae, rmse, mape

logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """Walk-forward expanding-window trainer for LightGBM.

    Trains using chronological folds with early stopping. Produces
    a serialized production model and a training report.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._feature_engineer = FeatureEngineer(self._settings)

    def _get_lgb_params(self) -> dict:
        """Get LightGBM hyperparameters from settings."""
        return {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": self._settings.num_leaves,
            "learning_rate": self._settings.learning_rate,
            "n_estimators": self._settings.n_estimators,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
        }

    def train(self, train_df: pd.DataFrame) -> dict:
        """Run walk-forward training pipeline.

        Args:
            train_df: Training DataFrame with Datetime and 6 region columns.

        Returns:
            Training report dictionary with fold metrics, feature importances,
            and best iteration counts.
        """
        logger.info("Starting walk-forward training on %d rows.", len(train_df))

        # Build feature matrix
        feature_df = self._feature_engineer.build_features(train_df)
        feature_names = self._feature_engineer.get_feature_names()

        X = feature_df[feature_names].values
        y = feature_df["target"].values
        n_samples = len(X)

        logger.info(
            "Feature matrix: %d samples × %d features.",
            n_samples,
            len(feature_names),
        )

        # Walk-forward fold boundaries
        n_folds = self._settings.walk_forward_folds  # 5
        fold_size = n_samples // n_folds

        fold_results: list[dict] = []
        best_iterations: list[int] = []

        # Folds 1-4: train + validate
        for fold_idx in range(1, n_folds):
            train_end = fold_size * fold_idx
            val_start = train_end
            val_end = fold_size * (fold_idx + 1)

            # Ensure last fold captures remaining rows
            if fold_idx == n_folds - 1:
                val_end = n_samples

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            logger.info(
                "Fold %d: Train [0:%d] (%d rows), Validate [%d:%d] (%d rows)",
                fold_idx,
                train_end,
                len(X_train),
                val_start,
                val_end,
                len(X_val),
            )

            params = self._get_lgb_params()
            model = lgb.LGBMRegressor(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self._settings.early_stopping_rounds
                    ),
                    lgb.log_evaluation(period=100),
                ],
            )

            best_iter = model.best_iteration_
            best_iterations.append(best_iter)

            # Predict on validation set
            y_pred = model.predict(X_val)

            fold_mae = mae(y_val, y_pred)
            fold_rmse = rmse(y_val, y_pred)
            fold_mape = mape(y_val, y_pred)

            fold_result = {
                "fold": fold_idx,
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "best_iteration": best_iter,
                "val_mae": fold_mae,
                "val_rmse": fold_rmse,
                "val_mape": fold_mape,
            }
            fold_results.append(fold_result)

            logger.info(
                "Fold %d results: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%, "
                "best_iter=%d",
                fold_idx,
                fold_mae,
                fold_rmse,
                fold_mape,
                best_iter,
            )

        # Compute average metrics across folds 1-4
        avg_mae = np.mean([r["val_mae"] for r in fold_results])
        avg_rmse = np.mean([r["val_rmse"] for r in fold_results])
        avg_mape = np.mean([r["val_mape"] for r in fold_results])
        avg_best_iter = int(np.mean(best_iterations))

        logger.info(
            "Average validation: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%, "
            "avg_best_iter=%d",
            avg_mae,
            avg_rmse,
            avg_mape,
            avg_best_iter,
        )

        # Fold 5: Train final production model on full corpus
        logger.info(
            "Fold 5: Training final model on all %d rows with "
            "n_estimators=%d.",
            n_samples,
            avg_best_iter,
        )

        final_params = self._get_lgb_params()
        final_params["n_estimators"] = avg_best_iter
        final_model = lgb.LGBMRegressor(**final_params)
        final_model.fit(X, y)

        # Feature importances
        importances = dict(
            zip(feature_names, final_model.feature_importances_.tolist())
        )

        # Serialize model
        model_path = Path(self._settings.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.booster_.save_model(str(model_path))
        logger.info("Final model saved to %s", model_path)

        # Build report
        report = {
            "n_training_rows": n_samples,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "fold_results": fold_results,
            "average_metrics": {
                "mae": avg_mae,
                "rmse": avg_rmse,
                "mape": avg_mape,
            },
            "average_best_iteration": avg_best_iter,
            "feature_importances": importances,
            "model_path": str(model_path),
        }

        # Save training report
        report_path = model_path.parent / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Training report saved to %s", report_path)

        return report
