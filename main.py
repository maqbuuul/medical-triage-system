"""
Enhanced main pipeline with comprehensive MLflow logging and error handling.
"""

import warnings
import joblib
import mlflow
from pathlib import Path
from src.data_preprocessing import preprocess_pipeline
from src.train_models import train_base_models, get_top_models
from src.tune_models import ModelTuner
from src.evaluate_models import evaluate_all_models
from src.config import MODELS_DIR, ARTIFACTS_DIR
from src.logging_config import setup_logging, get_logger
from src.mlflow_tracking import MLflowTracker

warnings.filterwarnings("ignore")


def main():
    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger("main")
    logger.info("🚀 Starting Enhanced Medical Triage ML Pipeline")

    # Initialize MLflow tracker
    tracker = MLflowTracker()

    try:
        # Create directories
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name="MEDICAL_TRIAGE_PIPELINE_COMPLETE"):
            # 1. Data preprocessing
            logger.info("📊 Starting data preprocessing...")
            data_dict = preprocess_pipeline()
            X_train, y_train = data_dict["X_train"], data_dict["y_train"]
            X_test, y_test = data_dict["X_test"], data_dict["y_test"]
            preprocessor = data_dict["preprocessor"]
            label_encoder = data_dict["label_encoder"]

            # Log dataset information
            tracker.log_dataset_info(X_train, X_test, y_train, y_test)
            logger.info(f"✅ Data loaded: Train: {X_train.shape}, Test: {X_test.shape}")

            # 2. Train base models with enhanced logging
            logger.info("🔨 Training base models...")
            accuracy_scores, trained_models = train_base_models(
                X_train, y_train, X_test, y_test
            )

            # Log base model results summary
            with mlflow.start_run(run_name="BASE_MODELS_SUMMARY", nested=True):
                best_base_accuracy = max(accuracy_scores.values())
                worst_base_accuracy = min(accuracy_scores.values())
                avg_base_accuracy = sum(accuracy_scores.values()) / len(accuracy_scores)

                mlflow.log_metric("best_base_model_accuracy", best_base_accuracy)
                mlflow.log_metric("worst_base_model_accuracy", worst_base_accuracy)
                mlflow.log_metric("average_base_model_accuracy", avg_base_accuracy)
                mlflow.log_param("total_base_models_trained", len(accuracy_scores))

                # Log individual base model scores
                for model_name, score in accuracy_scores.items():
                    mlflow.log_metric(f"base_{model_name}_accuracy", score)

            # 3. Get top models for tuning
            top_models = get_top_models(accuracy_scores, n_top=3)
            logger.info(f"🎯 Selected top models for tuning: {top_models}")

            with mlflow.start_run(run_name="MODEL_SELECTION", nested=True):
                mlflow.log_param("selected_models", ", ".join(top_models))
                mlflow.log_param("selection_criterion", "highest_accuracy")
                for i, model in enumerate(top_models):
                    mlflow.log_param(f"selected_model_{i + 1}", model)
                    mlflow.log_metric(
                        f"selected_model_{i + 1}_base_accuracy", accuracy_scores[model]
                    )

            # 4. Hyperparameter tuning with enhanced logging
            logger.info("⚙️ Starting hyperparameter tuning...")
            tuner = ModelTuner()

            # Track tuning start time
            import time

            tuning_start_time = time.time()

            best_models_dict, evaluation_results = tuner.tune_top_models(
                top_models,
                X_train,
                y_train,
                X_test,
                y_test,
                label_encoder,
                preprocessor,
            )

            tuning_duration = time.time() - tuning_start_time

            # Log tuning summary
            with mlflow.start_run(run_name="TUNING_SUMMARY", nested=True):
                mlflow.log_metric("total_tuning_time_seconds", tuning_duration)
                mlflow.log_metric("total_tuning_time_minutes", tuning_duration / 60)
                mlflow.log_param("tuning_method", "optuna")
                mlflow.log_param("models_tuned", len(best_models_dict))

                # Log tuning improvements
                for model_name, model_info in best_models_dict.items():
                    base_score = accuracy_scores.get(model_name, 0)
                    tuned_score = model_info.get("score", 0)
                    improvement = tuned_score - base_score

                    mlflow.log_metric(f"{model_name}_base_score", base_score)
                    mlflow.log_metric(f"{model_name}_tuned_score", tuned_score)
                    mlflow.log_metric(f"{model_name}_improvement", improvement)
                    mlflow.log_metric(
                        f"{model_name}_improvement_percentage",
                        (improvement / base_score * 100) if base_score > 0 else 0,
                    )

            # 5. Final model selection and saving
            best_model_name = max(
                best_models_dict.keys(), key=lambda k: best_models_dict[k]["score"]
            )
            best_model = best_models_dict[best_model_name]["model"]
            best_score = best_models_dict[best_model_name]["score"]

            logger.info(
                f"🏆 Best model selected: {best_model_name} (F1: {best_score:.4f})"
            )

            # Save models and artifacts
            final_model_path = MODELS_DIR / "final_model.pkl"
            encoder_path = MODELS_DIR / "label_encoder.pkl"
            preprocessor_path = MODELS_DIR / "preprocessor.pkl"

            joblib.dump(best_model, final_model_path)
            joblib.dump(label_encoder, encoder_path)
            joblib.dump(preprocessor, preprocessor_path)

            # Log artifacts to MLflow
            mlflow.log_artifact(str(final_model_path))
            mlflow.log_artifact(str(encoder_path))
            mlflow.log_artifact(str(preprocessor_path))

            logger.info(f"💾 Models saved:")
            logger.info(f"   Final model: {final_model_path}")
            logger.info(f"   Label encoder: {encoder_path}")
            logger.info(f"   Preprocessor: {preprocessor_path}")

            # 6. Log final production model with comprehensive information
            try:
                # Prepare model comparison data for final logging
                model_comparison_data = []
                for model_name, model_info in best_models_dict.items():
                    eval_result = evaluation_results.get(model_name, {})
                    metrics = eval_result.get("metrics", {})

                    model_comparison_data.append(
                        {
                            "Model": model_name,
                            "Accuracy": metrics.get("accuracy", 0.0),
                            "Precision": metrics.get("precision", 0.0),
                            "Recall": metrics.get("recall", 0.0),
                            "F1-Score": metrics.get("f1_score", 0.0),
                            "Validation Score": model_info.get("score", 0.0),
                        }
                    )

                final_metrics = tracker.log_final_model(
                    model_name=best_model_name,
                    model=best_model,
                    X_test=X_test,
                    y_test=y_test,
                    label_encoder=label_encoder,
                    artifacts_dir=ARTIFACTS_DIR,
                    model_comparison_data=model_comparison_data,
                )

                logger.info(
                    "✅ Final model logged to MLflow with comprehensive metrics"
                )

            except Exception as e:
                logger.warning(f"Final model MLflow logging encountered issues: {e}")

            # 7. Create experiment summary
            experiment_results = {}
            for model_name, model_info in best_models_dict.items():
                eval_result = evaluation_results.get(model_name, {})
                metrics = eval_result.get("metrics", {})

                experiment_results[model_name] = {
                    "accuracy": metrics.get("accuracy", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "validation_score": model_info.get("score", 0.0),
                    "base_accuracy": accuracy_scores.get(model_name, 0.0),
                }

            tracker.create_experiment_summary(experiment_results)

            # 8. Log final pipeline metrics
            pipeline_end_time = time.time()
            total_pipeline_time = (
                pipeline_end_time - tuning_start_time + tuning_duration
            )

            mlflow.log_metric(
                "PIPELINE_total_runtime_minutes", total_pipeline_time / 60
            )
            mlflow.log_metric("PIPELINE_final_best_f1_score", best_score)
            mlflow.log_metric(
                "PIPELINE_final_best_accuracy",
                experiment_results[best_model_name]["accuracy"],
            )
            mlflow.log_param("PIPELINE_final_best_model", best_model_name)
            mlflow.log_param("PIPELINE_status", "SUCCESS")
            mlflow.log_param("PIPELINE_models_evaluated", len(accuracy_scores))
            mlflow.log_param("PIPELINE_models_tuned", len(best_models_dict))

            # Calculate overall improvement
            best_base_score = accuracy_scores[best_model_name]
            overall_improvement = best_score - best_base_score
            mlflow.log_metric("PIPELINE_overall_improvement", overall_improvement)
            mlflow.log_metric(
                "PIPELINE_improvement_percentage",
                (overall_improvement / best_base_score * 100)
                if best_base_score > 0
                else 0,
            )

            # 9. Print comprehensive summary
            logger.info("=" * 80)
            logger.info("🎉 MEDICAL TRIAGE ML PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"🏆 FINAL RESULTS:")
            logger.info(f"   Best Model: {best_model_name}")
            logger.info(f"   Final F1-Score: {best_score:.4f}")
            logger.info(
                f"   Final Accuracy: {experiment_results[best_model_name]['accuracy']:.4f}"
            )
            logger.info(
                f"   Final Precision: {experiment_results[best_model_name]['precision']:.4f}"
            )
            logger.info(
                f"   Final Recall: {experiment_results[best_model_name]['recall']:.4f}"
            )
            logger.info(f"")
            logger.info(f"📈 PERFORMANCE IMPROVEMENTS:")
            logger.info(f"   Base Model Score: {best_base_score:.4f}")
            logger.info(f"   Tuned Model Score: {best_score:.4f}")
            logger.info(
                f"   Improvement: {overall_improvement:.4f} ({overall_improvement / best_base_score * 100:.1f}%)"
            )
            logger.info(f"")
            logger.info(f"⏱️  TIMING:")
            logger.info(
                f"   Total Pipeline Time: {total_pipeline_time / 60:.1f} minutes"
            )
            logger.info(f"   Tuning Time: {tuning_duration / 60:.1f} minutes")
            logger.info(f"")
            logger.info(f"📊 MODEL COMPARISON:")
            for model_name, results in experiment_results.items():
                logger.info(f"   {model_name}:")
                logger.info(f"     - Accuracy: {results['accuracy']:.4f}")
                logger.info(f"     - F1-Score: {results['f1_score']:.4f}")
                logger.info(f"     - Precision: {results['precision']:.4f}")
                logger.info(f"     - Recall: {results['recall']:.4f}")
            logger.info(f"")
            logger.info(f"💾 SAVED ARTIFACTS:")
            logger.info(f"   - Final model: {final_model_path}")
            logger.info(f"   - Label encoder: {encoder_path}")
            logger.info(f"   - Preprocessor: {preprocessor_path}")
            logger.info(f"   - Evaluation plots: {ARTIFACTS_DIR}/plots/")
            logger.info(f"")
            logger.info(f"📋 MLFLOW TRACKING:")
            logger.info(f"   - All runs logged to MLflow UI")
            logger.info(f"   - Production model registered")
            logger.info(f"   - Comprehensive metrics and artifacts saved")
            logger.info("=" * 80)

            return {
                "best_model": best_model,
                "best_model_name": best_model_name,
                "best_score": best_score,
                "experiment_results": experiment_results,
                "artifacts_saved": {
                    "model": str(final_model_path),
                    "encoder": str(encoder_path),
                    "preprocessor": str(preprocessor_path),
                },
            }

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")

        # Log failure to MLflow
        try:
            with mlflow.start_run(run_name="PIPELINE_FAILURE", nested=True):
                mlflow.log_param("PIPELINE_status", "FAILED")
                mlflow.log_param("error_message", str(e))
                mlflow.log_param("error_type", type(e).__name__)
        except:
            pass

        raise


if __name__ == "__main__":
    results = main()
    print(
        "\n🚀 Pipeline completed! Check MLflow UI for detailed results and comparisons."
    )
    print(
        f"Best model: {results['best_model_name']} with F1-Score: {results['best_score']:.4f}"
    )
