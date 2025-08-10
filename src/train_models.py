"""Training utilities for base models."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from src.utils.model_utils import get_model_instance
from src.config import CLASSIFIERS, CV_SPLITS, RND, N_JOBS, ARTIFACTS_DIR
from src.mlflow_tracking import MLflowTracker
from src.logging_config import get_logger
from src.config import OPTUNA_N_TRIALS

logger = get_logger("train_models")


def train_base_models(X_train, y_train, X_test, y_test):
    logger.info("Training base models...")
    tracker = MLflowTracker()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    accuracy_scores = {}
    trained_models = {}
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RND)

    for display_name, module_path, params in CLASSIFIERS:
        try:
            logger.info(f"Training {display_name}...")
            model = get_model_instance(module_path, params)

            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=N_JOBS
                )
            except Exception as e:
                logger.warning(
                    f"CV with n_jobs={N_JOBS} failed: {e}. Retrying with n_jobs=1"
                )
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=1
                )

            # Full training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)

            accuracy_scores[display_name] = test_accuracy
            trained_models[display_name] = model

            try:
                tracker.log_base_model(
                    model_name=display_name,
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    cv_scores=cv_scores,
                    params=params,
                )
            except Exception as e:
                logger.warning(f"MLflow logging failed for {display_name}: {e}")

            logger.info(f"✅ Trained {display_name} | Accuracy: {test_accuracy:.4f}")
        except Exception as e:
            logger.error(f"Failed to train {display_name}: {e}")

    # Save results
    results_df = pd.DataFrame(
        list(accuracy_scores.items()), columns=["Model", "Test_Accuracy"]
    )
    results_path = ARTIFACTS_DIR / "base_model_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved base model results to {results_path}")

    return accuracy_scores, trained_models


def get_top_models(accuracy_scores: dict, n_top: int = 3):
    sorted_models = sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True)
    top_models = [name for name, _ in sorted_models[:n_top]]

    # Adjust based on Optuna capacity
    if len(top_models) > 3 and OPTUNA_N_TRIALS > 30:
        top_models = top_models[:3]
    logger.info(f"Selected models for tuning: {top_models}")

    return top_models
