"""Improved comprehensive model evaluation module with better MLflow integration."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
import joblib
import warnings
from src.config import ARTIFACTS_DIR, FEATURE_GROUPS
from src.logging_config import get_logger

logger = get_logger("evaluate_models")
warnings.filterwarnings("ignore")


def create_feature_names_fallback(n_features):
    """Create fallback feature names when preprocessor info is not available."""
    feature_names = []

    # Add ordinal features
    ordinal_features = FEATURE_GROUPS.get("ordinal", [])
    feature_names.extend(ordinal_features)

    # Add binary features
    binary_features = FEATURE_GROUPS.get("binary", [])
    feature_names.extend(binary_features)

    # Add nominal features (simplified)
    nominal_features = FEATURE_GROUPS.get("nominal", [])
    for feature in nominal_features:
        feature_names.append(f"{feature}_encoded")

    # If we still don't have enough features, add generic ones
    while len(feature_names) < n_features:
        feature_names.append(f"feature_{len(feature_names)}")

    # If we have too many, truncate
    return feature_names[:n_features]


class ModelEvaluator:
    def __init__(self, output_dir: Path = ARTIFACTS_DIR):
        self.output_dir = output_dir
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def evaluate_single_model(
        self, model, model_name, X_test, y_test, label_encoder, feature_names=None
    ):
        logger.info(f"Evaluating {model_name}...")

        # Start MLflow run for detailed evaluation
        with mlflow.start_run(run_name=f"evaluation_{model_name}", nested=True):
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = (
                    model.predict_proba(X_test)
                    if hasattr(model, "predict_proba")
                    else None
                )

                class_names = (
                    label_encoder.classes_
                    if hasattr(label_encoder, "classes_")
                    else None
                )

                # Calculate all metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="weighted"
                )

                # Log metrics to MLflow
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1_score", f1)

                # Calculate per-class metrics
                report_dict = classification_report(
                    y_test, y_pred, target_names=class_names, output_dict=True
                )

                # Log per-class metrics
                if class_names is not None:
                    for class_name in class_names:
                        if class_name in report_dict:
                            class_metrics = report_dict[class_name]
                            mlflow.log_metric(
                                f"{class_name}_precision", class_metrics["precision"]
                            )
                            mlflow.log_metric(
                                f"{class_name}_recall", class_metrics["recall"]
                            )
                            mlflow.log_metric(
                                f"{class_name}_f1", class_metrics["f1-score"]
                            )
                            mlflow.log_metric(
                                f"{class_name}_support", class_metrics["support"]
                            )

                # Calculate AUC if possible
                if y_pred_proba is not None:
                    try:
                        n_classes = len(np.unique(y_test))
                        if n_classes == 2:
                            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc_score = roc_auc_score(
                                y_test,
                                y_pred_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                        mlflow.log_metric("test_auc", auc_score)
                        logger.info(f"{model_name} AUC: {auc_score:.4f}")
                    except Exception as e:
                        logger.warning(f"AUC calculation failed for {model_name}: {e}")

                # Create visualizations
                cm = confusion_matrix(y_test, y_pred)
                plot_paths = {
                    "confusion_matrix": self._plot_cm(cm, class_names, model_name)
                }

                # ROC curves with fixed error handling
                if y_pred_proba is not None:
                    try:
                        plot_paths["roc_curve"] = self._plot_roc_curve_fixed(
                            y_test, y_pred_proba, class_names, model_name
                        )
                    except Exception as e:
                        logger.error(f"ROC curve failed for {model_name}: {e}")

                    try:
                        plot_paths["precision_recall"] = (
                            self._plot_precision_recall_curve_fixed(
                                y_test, y_pred_proba, class_names, model_name
                            )
                        )
                    except Exception as e:
                        logger.error(
                            f"Precision-recall curve failed for {model_name}: {e}"
                        )

                # Feature importance
                try:
                    if hasattr(model, "feature_importances_"):
                        if feature_names is None:
                            n_features = len(model.feature_importances_)
                            feature_names = create_feature_names_fallback(n_features)

                        plot_paths["feature_importance"] = (
                            self._plot_feature_importance(
                                model, feature_names, model_name
                            )
                        )

                        # Log top 10 feature importances to MLflow
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1][:10]
                        for i, idx in enumerate(indices):
                            if idx < len(feature_names):
                                mlflow.log_metric(
                                    f"feature_importance_{i + 1}_{feature_names[idx]}",
                                    importances[idx],
                                )

                except Exception as e:
                    logger.error(
                        f"Feature importance plot failed for {model_name}: {e}"
                    )

                # Log plots as artifacts
                for plot_name, plot_path in plot_paths.items():
                    if plot_path and plot_path.exists():
                        mlflow.log_artifact(str(plot_path), f"plots/{plot_name}")

                # SHAP analysis (simplified to avoid errors)
                try:
                    if feature_names is not None and X_test.shape[0] > 0:
                        shap_plots = self._generate_shap_plots(
                            model, X_test, feature_names, model_name
                        )
                        plot_paths.update(shap_plots)

                        # Log SHAP plots
                        for shap_plot_name, shap_plot_path in shap_plots.items():
                            if shap_plot_path and shap_plot_path.exists():
                                mlflow.log_artifact(str(shap_plot_path), "plots/shap")

                except Exception as e:
                    logger.error(f"SHAP plots failed for {model_name}: {e}")

                # Save detailed results
                detailed_results = {
                    "model_name": model_name,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                    "class_names": class_names,
                    "classification_report": report_dict,
                    "confusion_matrix": cm,
                    "plot_paths": plot_paths,
                    "metrics": {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    },
                }

                self._save_detailed_results(detailed_results, model_name)

                # Log model performance summary
                logger.info(f"✅ {model_name} Evaluation Complete:")
                logger.info(f"   Accuracy: {accuracy:.4f}")
                logger.info(f"   Precision: {precision:.4f}")
                logger.info(f"   Recall: {recall:.4f}")
                logger.info(f"   F1-Score: {f1:.4f}")

                return {
                    "model_name": model_name,
                    "report": report_dict,
                    "confusion_matrix": cm,
                    "plot_paths": plot_paths,
                    "metrics": detailed_results["metrics"],
                }

            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}: {e}")
                mlflow.log_param("evaluation_error", str(e))
                return None

    def _plot_cm(self, cm: np.ndarray, class_names: list, model_name: str):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xticks(rotation=45)

        plot_path = self.plots_dir / f"{model_name}_confusion_matrix.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return plot_path

    def _plot_roc_curve_fixed(self, y_true, y_pred_proba, class_names, model_name):
        """Fixed ROC curve plotting."""
        n_classes = y_pred_proba.shape[1]

        # Handle binary classification differently
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - {model_name}")
            ax.legend(loc="lower right")

        else:
            # Multi-class ROC
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

                class_label = class_names[i] if class_names else f"Class {i}"
                ax.plot(
                    fpr[i],
                    tpr[i],
                    color=colors[i],
                    lw=2,
                    label=f"{class_label} (AUC = {roc_auc[i]:.2f})",
                )

            ax.plot([0, 1], [0, 1], "k--", lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"Multi-class ROC Curve - {model_name}")
            ax.legend(loc="lower right")

        plot_path = self.plots_dir / f"{model_name}_roc_curve.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return plot_path

    def _plot_precision_recall_curve_fixed(
        self, y_true, y_pred_proba, class_names, model_name
    ):
        """Fixed precision-recall curve plotting."""
        n_classes = y_pred_proba.shape[1]

        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                recall,
                precision,
                color="darkorange",
                lw=2,
                label=f"Precision-Recall curve (AP = {avg_precision:.2f})",
            )
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curve - {model_name}")
            ax.legend(loc="best")

        else:
            # Multi-class precision-recall
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(
                    y_bin[:, i], y_pred_proba[:, i]
                )
                avg_precision = average_precision_score(y_bin[:, i], y_pred_proba[:, i])

                class_label = class_names[i] if class_names else f"Class {i}"
                ax.plot(
                    recall,
                    precision,
                    color=colors[i],
                    lw=2,
                    label=f"{class_label} (AP = {avg_precision:.2f})",
                )

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Multi-class Precision-Recall Curve - {model_name}")
            ax.legend(loc="best")

        plot_path = self.plots_dir / f"{model_name}_precision_recall.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return plot_path

    def _plot_feature_importance(self, model, feature_names, model_name):
        if not hasattr(model, "feature_importances_"):
            return None

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Ensure we have enough feature names
        if len(feature_names) < len(importances):
            for i in range(len(feature_names), len(importances)):
                feature_names.append(f"feature_{i}")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Show top 20 features to avoid overcrowding
        n_features = min(20, len(indices))
        indices = indices[:n_features]

        ax.barh(range(n_features), importances[indices], align="center")
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel("Relative Importance")
        ax.set_title(f"Top {n_features} Feature Importances - {model_name}")

        plot_path = self.plots_dir / f"{model_name}_feature_importance.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return plot_path

    def _generate_shap_plots(self, model, X_test, feature_names, model_name):
        """Generate SHAP plots with better error handling."""
        try:
            # Use smaller sample for SHAP
            sample_size = min(50, X_test.shape[0])
            X_shap = X_test[:sample_size]

            plot_paths = {}

            # Try different explainer types based on model
            if hasattr(model, "tree_") or hasattr(model, "estimators_"):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_shap)
                except:
                    # Fallback to permutation explainer
                    explainer = shap.Explainer(model.predict, X_shap[:20])
                    shap_values = explainer(X_shap)
            else:
                explainer = shap.Explainer(model.predict, X_shap[:20])
                shap_values = explainer(X_shap)

            # Summary plot
            try:
                plt.figure(figsize=(10, 6))
                if isinstance(shap_values, list):
                    # Multi-class - just plot first class
                    shap.summary_plot(
                        shap_values[0], X_shap, feature_names=feature_names, show=False
                    )
                else:
                    shap.summary_plot(
                        shap_values, X_shap, feature_names=feature_names, show=False
                    )

                plot_path = self.plots_dir / f"{model_name}_shap_summary.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                plt.close()
                plot_paths["shap_summary"] = plot_path
            except Exception as e:
                logger.warning(f"SHAP summary plot failed for {model_name}: {e}")

            return plot_paths

        except Exception as e:
            logger.warning(f"SHAP analysis failed for {model_name}: {e}")
            return {}

    def _save_detailed_results(self, results, model_name):
        results_path = self.output_dir / f"{model_name}_evaluation_results.joblib"
        joblib.dump(results, results_path)
        logger.info(f"Saved detailed results for {model_name}")

    def compare_models(self, evaluation_results):
        """Create model comparison with MLflow logging."""
        comparison_data = []

        with mlflow.start_run(run_name="model_comparison", nested=True):
            for model_name, results in evaluation_results.items():
                if results is None:
                    continue

                metrics = results.get("metrics", {})
                validation_score = results.get("validation_score", 0.0)

                comparison_data.append(
                    {
                        "Model": model_name,
                        "Accuracy": metrics.get("accuracy", 0.0),
                        "Precision": metrics.get("precision", 0.0),
                        "Recall": metrics.get("recall", 0.0),
                        "F1-Score": metrics.get("f1_score", 0.0),
                        "Validation Score": validation_score,
                    }
                )

                # Log individual model metrics to comparison run
                mlflow.log_metric(
                    f"{model_name}_accuracy", metrics.get("accuracy", 0.0)
                )
                mlflow.log_metric(
                    f"{model_name}_precision", metrics.get("precision", 0.0)
                )
                mlflow.log_metric(f"{model_name}_recall", metrics.get("recall", 0.0))
                mlflow.log_metric(
                    f"{model_name}_f1_score", metrics.get("f1_score", 0.0)
                )

            if comparison_data:
                df = pd.DataFrame(comparison_data)
                df = df.sort_values("F1-Score", ascending=False)

                # Save comparison table
                table_path = self.output_dir / "model_comparison.csv"
                df.to_csv(table_path, index=False)
                mlflow.log_artifact(str(table_path))

                # Log best model metrics
                best_model = df.iloc[0]
                mlflow.log_metric("best_model_accuracy", best_model["Accuracy"])
                mlflow.log_metric("best_model_f1_score", best_model["F1-Score"])
                mlflow.log_param("best_model_name", best_model["Model"])

                # Create comparison plot
                fig, ax = plt.subplots(figsize=(14, 8))
                metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]

                x = np.arange(len(df))
                width = 0.2

                for i, metric in enumerate(metrics_to_plot):
                    ax.bar(x + i * width, df[metric], width, label=metric)

                ax.set_xlabel("Models")
                ax.set_ylabel("Score")
                ax.set_title("Model Performance Comparison")
                ax.set_xticks(x + width * 1.5)
                ax.set_xticklabels(df["Model"], rotation=45)
                ax.legend()
                ax.set_ylim([0, 1])

                # Add value labels on bars
                for i, metric in enumerate(metrics_to_plot):
                    for j, value in enumerate(df[metric]):
                        ax.text(
                            j + i * width,
                            value + 0.01,
                            f"{value:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                plot_path = self.plots_dir / "model_comparison.png"
                fig.savefig(plot_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
                mlflow.log_artifact(str(plot_path))

                logger.info(
                    f"✅ Model comparison completed. Best model: {best_model['Model']}"
                )


def evaluate_all_models(
    best_models_dict, X_test, y_test, label_encoder, preprocessor=None
):
    """Enhanced model evaluation with comprehensive MLflow logging."""
    logger.info("Starting comprehensive model evaluation...")

    evaluator = ModelEvaluator()

    # Create feature names with fallback
    feature_names = None
    if preprocessor is not None:
        try:
            from src.utils.data_utils import create_feature_names

            feature_names = create_feature_names(preprocessor)
        except Exception as e:
            logger.warning(f"Could not create feature names from preprocessor: {e}")

    # If we still don't have feature names, create fallback
    if feature_names is None and X_test.shape[1] > 0:
        feature_names = create_feature_names_fallback(X_test.shape[1])
        logger.info("Using fallback feature names")

    all_evaluation_results = {}

    # Evaluate each model
    for model_name, model_info in best_models_dict.items():
        if "model" not in model_info:
            logger.error(f"Invalid model info for {model_name}")
            continue

        eval_results = evaluator.evaluate_single_model(
            model=model_info["model"],
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
            feature_names=feature_names,
        )

        if eval_results:
            eval_results["validation_score"] = model_info.get("score", 0.0)
            all_evaluation_results[model_name] = eval_results

    # Create comprehensive comparison
    evaluator.compare_models(all_evaluation_results)

    logger.info("✅ Comprehensive evaluation completed")
    return all_evaluation_results
