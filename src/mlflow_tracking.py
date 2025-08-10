"""
Enhanced MLflow tracking utilities for medical triage ML pipeline.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.logging_config import get_logger

logger = get_logger("mlflow_tracking")


class MLflowTracker:
    """Enhanced MLflow tracking wrapper for medical triage pipeline."""

    def __init__(self, experiment_name: str = MLFLOW_EXPERIMENT_NAME):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.setup_experiment()

    def setup_experiment(self):
        """Set up MLflow experiment and tracking URI."""
        try:
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment set to: {self.experiment_name}")

        except Exception as e:
            logger.error(f"Failed to set up MLflow experiment: {e}")
            raise

    def log_base_model(self, model_name, model, X_test, y_test, cv_scores, params=None):
        """Log base model run to MLflow with comprehensive metrics."""
        with mlflow.start_run(run_name=f"base_{model_name}"):
            try:
                # Log parameters
                if params:
                    mlflow.log_params(params)

                # Log model type and info
                mlflow.log_param("model_type", "base_model")
                mlflow.log_param("model_name", model_name)

                # Predictions and probabilities
                y_pred = model.predict(X_test)
                y_pred_proba = (
                    model.predict_proba(X_test)
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Basic metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test, y_pred, average="weighted"
                )

                # Log metrics
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1_score", f1)
                mlflow.log_metric("cv_mean_score", cv_scores.mean())
                mlflow.log_metric("cv_std_score", cv_scores.std())

                # Log per-fold CV scores
                for i, score in enumerate(cv_scores):
                    mlflow.log_metric(f"cv_fold_{i + 1}_score", score)

                # AUC score
                if y_pred_proba is not None:
                    try:
                        n_classes = len(np.unique(y_test))
                        if n_classes == 2:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = roc_auc_score(
                                y_test,
                                y_pred_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                        mlflow.log_metric("test_auc", auc)
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC for {model_name}: {e}")

                # Per-class metrics
                try:
                    report = classification_report(y_test, y_pred, output_dict=True)
                    class_names = [str(i) for i in np.unique(y_test)]

                    for class_name in class_names:
                        if class_name in report:
                            class_metrics = report[class_name]
                            mlflow.log_metric(
                                f"class_{class_name}_precision",
                                class_metrics["precision"],
                            )
                            mlflow.log_metric(
                                f"class_{class_name}_recall", class_metrics["recall"]
                            )
                            mlflow.log_metric(
                                f"class_{class_name}_f1", class_metrics["f1-score"]
                            )
                            mlflow.log_metric(
                                f"class_{class_name}_support", class_metrics["support"]
                            )

                except Exception as e:
                    logger.warning(
                        f"Could not log per-class metrics for {model_name}: {e}"
                    )

                # Feature importance (if available)
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    # Log top 10 feature importances
                    top_indices = np.argsort(importances)[::-1][:10]
                    for i, idx in enumerate(top_indices):
                        mlflow.log_metric(
                            f"feature_importance_rank_{i + 1}", importances[idx]
                        )

                    # Log feature importance statistics
                    mlflow.log_metric("feature_importance_mean", np.mean(importances))
                    mlflow.log_metric("feature_importance_std", np.std(importances))
                    mlflow.log_metric("feature_importance_max", np.max(importances))

                # Confusion matrix statistics
                cm = confusion_matrix(y_test, y_pred)
                mlflow.log_metric("confusion_matrix_trace", np.trace(cm))
                mlflow.log_metric("total_predictions", len(y_test))

                # Log model
                if "xgb" in model_name.lower():
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")

                logger.info(
                    f"Logged base model: {model_name} (accuracy: {accuracy:.4f}, f1: {f1:.4f})"
                )

                return {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                }

            except Exception as e:
                logger.error(f"Failed to log base model {model_name}: {e}")
                raise

    def log_tuning_run(
        self,
        model_name,
        best_model,
        best_params,
        best_score,
        X_test,
        y_test,
        tuning_method="random",
        study_info=None,
    ):
        """Log hyperparameter tuning run to MLflow with comprehensive information."""
        with mlflow.start_run(run_name=f"tuned_{model_name}_{tuning_method}"):
            try:
                # Log tuning parameters
                mlflow.log_params(best_params)
                mlflow.log_param("tuning_method", tuning_method)
                mlflow.log_param("model_type", "tuned_model")
                mlflow.log_param("model_name", model_name)

                # Log study information if provided
                if study_info:
                    mlflow.log_param("n_trials", study_info.get("n_trials", 0))
                    mlflow.log_param(
                        "study_direction", study_info.get("direction", "maximize")
                    )
                    if "best_trial" in study_info:
                        mlflow.log_metric(
                            "best_trial_number", study_info["best_trial"].number
                        )
                        mlflow.log_metric(
                            "best_trial_value", study_info["best_trial"].value
                        )

                # Predictions and metrics
                y_pred = best_model.predict(X_test)
                y_pred_proba = (
                    best_model.predict_proba(X_test)
                    if hasattr(best_model, "predict_proba")
                    else None
                )

                # Comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test, y_pred, average="weighted"
                )

                # Log all metrics
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1_score", f1)
                mlflow.log_metric("cv_best_score", best_score)

                # Calculate improvement over baseline (if available)
                mlflow.log_metric(
                    "performance_improvement", f1 - best_score if best_score > 0 else 0
                )

                # AUC and other advanced metrics
                if y_pred_proba is not None:
                    try:
                        n_classes = len(np.unique(y_test))
                        if n_classes == 2:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = roc_auc_score(
                                y_test,
                                y_pred_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                        mlflow.log_metric("test_auc", auc)
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC: {e}")

                # Per-class detailed metrics
                try:
                    report = classification_report(y_test, y_pred, output_dict=True)
                    class_names = [str(i) for i in np.unique(y_test)]

                    for class_name in class_names:
                        if class_name in report:
                            class_metrics = report[class_name]
                            mlflow.log_metric(
                                f"class_{class_name}_precision",
                                class_metrics["precision"],
                            )
                            mlflow.log_metric(
                                f"class_{class_name}_recall", class_metrics["recall"]
                            )
                            mlflow.log_metric(
                                f"class_{class_name}_f1", class_metrics["f1-score"]
                            )
                            mlflow.log_metric(
                                f"class_{class_name}_support", class_metrics["support"]
                            )

                    # Log macro and micro averages
                    if "macro avg" in report:
                        mlflow.log_metric(
                            "macro_avg_precision", report["macro avg"]["precision"]
                        )
                        mlflow.log_metric(
                            "macro_avg_recall", report["macro avg"]["recall"]
                        )
                        mlflow.log_metric(
                            "macro_avg_f1", report["macro avg"]["f1-score"]
                        )

                    if "micro avg" in report:
                        mlflow.log_metric(
                            "micro_avg_precision", report["micro avg"]["precision"]
                        )
                        mlflow.log_metric(
                            "micro_avg_recall", report["micro avg"]["recall"]
                        )
                        mlflow.log_metric(
                            "micro_avg_f1", report["micro avg"]["f1-score"]
                        )

                except Exception as e:
                    logger.warning(f"Could not log detailed per-class metrics: {e}")

                # Feature importance for tree-based models
                if hasattr(best_model, "feature_importances_"):
                    importances = best_model.feature_importances_
                    # Log comprehensive feature importance statistics
                    mlflow.log_metric("n_features", len(importances))
                    mlflow.log_metric("feature_importance_mean", np.mean(importances))
                    mlflow.log_metric("feature_importance_std", np.std(importances))
                    mlflow.log_metric("feature_importance_max", np.max(importances))
                    mlflow.log_metric("feature_importance_min", np.min(importances))

                    # Log top features
                    top_indices = np.argsort(importances)[::-1][:15]
                    for i, idx in enumerate(top_indices):
                        mlflow.log_metric(
                            f"top_feature_{i + 1}_importance", importances[idx]
                        )

                # Model complexity metrics
                try:
                    if hasattr(best_model, "n_estimators"):
                        mlflow.log_param("n_estimators_used", best_model.n_estimators)
                    if hasattr(best_model, "max_depth"):
                        mlflow.log_param("max_depth_used", best_model.max_depth)
                    if hasattr(best_model, "n_features_in_"):
                        mlflow.log_param("n_features_in", best_model.n_features_in_)
                except Exception as e:
                    logger.warning(f"Could not log model complexity metrics: {e}")

                # Confusion matrix analysis
                cm = confusion_matrix(y_test, y_pred)
                mlflow.log_metric("confusion_matrix_trace", np.trace(cm))
                mlflow.log_metric("total_correct_predictions", np.trace(cm))
                mlflow.log_metric("total_predictions", np.sum(cm))

                # Calculate and log class balance
                class_distribution = np.bincount(y_test)
                mlflow.log_metric(
                    "class_balance_ratio",
                    np.max(class_distribution) / np.min(class_distribution),
                )

                # Log model
                if "xgb" in model_name.lower():
                    mlflow.xgboost.log_model(best_model, "model")
                else:
                    mlflow.sklearn.log_model(best_model, "model")

                logger.info(
                    f"Logged tuned model: {model_name} (accuracy: {accuracy:.4f}, f1: {f1:.4f}, method: {tuning_method})"
                )

                return {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "best_cv_score": best_score,
                }

            except Exception as e:
                logger.error(f"Failed to log tuned model {model_name}: {e}")
                raise

    def log_final_model(
        self,
        model_name,
        model,
        X_test,
        y_test,
        label_encoder,
        artifacts_dir: Path = None,
        model_comparison_data=None,
    ):
        """Log final production model with comprehensive artifacts and metadata."""
        with mlflow.start_run(run_name=f"PRODUCTION_{model_name}_final"):
            try:
                # Log model metadata
                mlflow.log_param("model_type", "production_model")
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("deployment_ready", True)

                # Predictions and probabilities
                y_pred = model.predict(X_test)
                y_pred_proba = (
                    model.predict_proba(X_test)
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test, y_pred, average="weighted"
                )

                # Log primary metrics
                mlflow.log_metric("FINAL_accuracy", accuracy)
                mlflow.log_metric("FINAL_precision", precision)
                mlflow.log_metric("FINAL_recall", recall)
                mlflow.log_metric("FINAL_f1_score", f1)

                # AUC calculation
                if y_pred_proba is not None:
                    try:
                        n_classes = len(np.unique(y_test))
                        if n_classes == 2:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = roc_auc_score(
                                y_test,
                                y_pred_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                        mlflow.log_metric("FINAL_auc", auc)
                    except Exception as e:
                        logger.warning(f"Could not calculate final AUC: {e}")

                # Detailed classification report
                target_names = (
                    label_encoder.classes_
                    if hasattr(label_encoder, "classes_")
                    else None
                )
                report = classification_report(
                    y_test, y_pred, target_names=target_names, output_dict=True
                )

                # Log detailed per-class metrics for production
                if target_names is not None:
                    for class_name in target_names:
                        if str(class_name) in report:
                            class_metrics = report[str(class_name)]
                            mlflow.log_metric(
                                f"PRODUCTION_{class_name}_precision",
                                class_metrics["precision"],
                            )
                            mlflow.log_metric(
                                f"PRODUCTION_{class_name}_recall",
                                class_metrics["recall"],
                            )
                            mlflow.log_metric(
                                f"PRODUCTION_{class_name}_f1", class_metrics["f1-score"]
                            )
                            mlflow.log_metric(
                                f"PRODUCTION_{class_name}_support",
                                class_metrics["support"],
                            )

                # Save and log classification report
                report_df = pd.DataFrame(report).transpose()
                report_path = "classification_report.csv"
                report_df.to_csv(report_path)
                mlflow.log_artifact(report_path)

                # Log confusion matrix data
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"True_{i}" for i in range(len(cm))],
                    columns=[f"Pred_{i}" for i in range(len(cm))],
                )
                cm_path = "confusion_matrix.csv"
                cm_df.to_csv(cm_path)
                mlflow.log_artifact(cm_path)

                # Model comparison data
                if model_comparison_data:
                    comparison_df = pd.DataFrame(model_comparison_data)
                    comparison_path = "model_comparison_final.csv"
                    comparison_df.to_csv(comparison_path, index=False)
                    mlflow.log_artifact(comparison_path)

                    # Log why this model was selected
                    best_metric = comparison_df[comparison_df["Model"] == model_name]
                    if not best_metric.empty:
                        mlflow.log_metric(
                            "selection_f1_score", best_metric["F1-Score"].iloc[0]
                        )
                        mlflow.log_metric(
                            "selection_accuracy", best_metric["Accuracy"].iloc[0]
                        )

                # Feature importance for final model
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_

                    # Create feature importance plot
                    plt.figure(figsize=(12, 8))
                    indices = np.argsort(importances)[::-1][:20]
                    plt.bar(range(20), importances[indices])
                    plt.title(f"Top 20 Feature Importances - {model_name}")
                    plt.xlabel("Feature Rank")
                    plt.ylabel("Importance")

                    importance_plot_path = "final_feature_importance.png"
                    plt.savefig(importance_plot_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    mlflow.log_artifact(importance_plot_path)

                    # Log numerical importance data
                    importance_df = pd.DataFrame(
                        {
                            "feature_index": range(len(importances)),
                            "importance": importances,
                            "rank": np.argsort(-importances) + 1,
                        }
                    ).sort_values("importance", ascending=False)

                    importance_csv_path = "feature_importances.csv"
                    importance_df.to_csv(importance_csv_path, index=False)
                    mlflow.log_artifact(importance_csv_path)

                # Log model with registry
                if "xgb" in model_name.lower():
                    mlflow.xgboost.log_model(
                        model, "model", registered_model_name=f"{model_name}_production"
                    )
                else:
                    mlflow.sklearn.log_model(
                        model, "model", registered_model_name=f"{model_name}_production"
                    )

                # Log artifacts directory
                if artifacts_dir and artifacts_dir.exists():
                    for artifact_file in artifacts_dir.glob("*"):
                        if artifact_file.is_file():
                            try:
                                mlflow.log_artifact(str(artifact_file))
                            except Exception as e:
                                logger.warning(
                                    f"Could not log artifact {artifact_file}: {e}"
                                )

                # Log label encoder
                encoder_path = "label_encoder.pkl"
                joblib.dump(label_encoder, encoder_path)
                mlflow.log_artifact(encoder_path)

                # Log deployment metadata
                mlflow.log_param("ready_for_deployment", True)
                mlflow.log_param("model_version", "1.0.0")
                mlflow.log_param("training_completed", True)

                logger.info(f"✅ Logged FINAL production model: {model_name}")
                logger.info(f"   Final Accuracy: {accuracy:.4f}")
                logger.info(f"   Final F1-Score: {f1:.4f}")
                logger.info(f"   Final Precision: {precision:.4f}")
                logger.info(f"   Final Recall: {recall:.4f}")

                return {
                    "final_accuracy": accuracy,
                    "final_f1_score": f1,
                    "final_precision": precision,
                    "final_recall": recall,
                }

            except Exception as e:
                logger.error(f"Failed to log final model {model_name}: {e}")
                raise

    def log_dataset_info(self, X_train, X_test, y_train, y_test):
        """Log comprehensive dataset information."""
        with mlflow.start_run(run_name="dataset_info", nested=True):
            try:
                # Dataset shapes
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples", X_test.shape[0])
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("total_samples", X_train.shape[0] + X_test.shape[0])

                # Class distribution
                unique_train, counts_train = np.unique(y_train, return_counts=True)
                unique_test, counts_test = np.unique(y_test, return_counts=True)

                mlflow.log_param("n_classes", len(unique_train))

                # Log class distributions
                for i, (class_label, count) in enumerate(
                    zip(unique_train, counts_train)
                ):
                    mlflow.log_metric(f"train_class_{class_label}_count", count)
                    mlflow.log_metric(
                        f"train_class_{class_label}_percentage",
                        count / len(y_train) * 100,
                    )

                for i, (class_label, count) in enumerate(zip(unique_test, counts_test)):
                    mlflow.log_metric(f"test_class_{class_label}_count", count)
                    mlflow.log_metric(
                        f"test_class_{class_label}_percentage",
                        count / len(y_test) * 100,
                    )

                # Class imbalance metrics
                mlflow.log_metric(
                    "train_class_imbalance_ratio", max(counts_train) / min(counts_train)
                )
                mlflow.log_metric(
                    "test_class_imbalance_ratio", max(counts_test) / min(counts_test)
                )

                logger.info("✅ Dataset information logged to MLflow")

            except Exception as e:
                logger.error(f"Failed to log dataset info: {e}")

    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Enhanced artifact logging."""
        try:
            if artifact_name:
                mlflow.log_artifact(artifact_path, artifact_name)
            else:
                mlflow.log_artifact(artifact_path)
            logger.debug(f"Logged artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact {artifact_path}: {e}")

    def log_metric(self, key: str, value: float, step: int = None):
        """Enhanced metric logging."""
        try:
            if step is not None:
                mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metric(key, value)
            logger.debug(f"Logged metric: {key} = {value}")
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")

    def log_param(self, key: str, value: Any):
        """Enhanced parameter logging."""
        try:
            mlflow.log_param(key, value)
            logger.debug(f"Logged parameter: {key} = {value}")
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")

    def create_experiment_summary(self, experiment_results):
        """Create and log a comprehensive experiment summary."""
        with mlflow.start_run(run_name="EXPERIMENT_SUMMARY"):
            try:
                # Log experiment-wide statistics
                mlflow.log_param("experiment_type", "medical_triage_classification")
                mlflow.log_param("total_models_evaluated", len(experiment_results))

                # Find best performing model
                best_f1 = 0
                best_model_name = ""
                for model_name, results in experiment_results.items():
                    if results and "f1_score" in results:
                        if results["f1_score"] > best_f1:
                            best_f1 = results["f1_score"]
                            best_model_name = model_name

                mlflow.log_param("best_model_overall", best_model_name)
                mlflow.log_metric("best_f1_score_achieved", best_f1)

                # Log summary statistics
                f1_scores = [
                    r.get("f1_score", 0) for r in experiment_results.values() if r
                ]
                if f1_scores:
                    mlflow.log_metric("f1_score_mean", np.mean(f1_scores))
                    mlflow.log_metric("f1_score_std", np.std(f1_scores))
                    mlflow.log_metric("f1_score_range", max(f1_scores) - min(f1_scores))

                logger.info("✅ Experiment summary logged to MLflow")

            except Exception as e:
                logger.error(f"Failed to log experiment summary: {e}")
