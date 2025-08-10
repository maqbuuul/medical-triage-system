"""Enhanced model hyperparameter tuning with comprehensive MLflow logging."""

import time
import optuna
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from src.utils.model_utils import get_model_instance
from src.config import (
    TUNING_CONFIGS,
    TUNING_CV_SPLITS,
    RND,
    N_JOBS,
    ARTIFACTS_DIR,
    CLASSIFIERS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
)
from src.mlflow_tracking import MLflowTracker
from src.logging_config import get_logger

logger = get_logger("tune_models")


def _get_classifier_entry(display_name: str):
    return next((entry for entry in CLASSIFIERS if entry[0] == display_name), None)


class ModelTuner:
    def __init__(self):
        self.tracker = MLflowTracker()
        self.scorer = make_scorer(f1_score, average="weighted")

    def objective(self, trial, display_name, X, y):
        entry = _get_classifier_entry(display_name)
        if not entry:
            raise ValueError(f"Classifier entry not found for {display_name}")

        _, module_path, base_params = entry
        param_space = TUNING_CONFIGS.get(display_name, {})

        params = base_params.copy()
        for param_name, values in param_space.items():
            if isinstance(values, list):
                if all(isinstance(v, (int, float)) for v in values):
                    params[param_name] = trial.suggest_categorical(param_name, values)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, values)

        # Special handling for XGBoost
        if "XGBoost" in display_name:
            params.setdefault("tree_method", "hist")
            params.setdefault("device", "cpu")

        try:
            model = get_model_instance(module_path, params)
            cv = StratifiedKFold(
                n_splits=TUNING_CV_SPLITS, shuffle=True, random_state=RND
            )
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=self.scorer, n_jobs=N_JOBS
            )
            return np.mean(scores)
        except Exception as e:
            logger.warning(f"Trial failed for {display_name}: {e}")
            return 0.0  # Return poor score for failed trials

    def tune_with_optuna(self, display_name, X_train, y_train, X_test, y_test):
        logger.info(f"🎯 Starting Optuna tuning for {display_name}...")

        def objective_wrapper(trial):
            return self.objective(trial, display_name, X_train, y_train)

        # Create study with more robust settings
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RND, n_startup_trials=10),
        )

        start_time = time.time()

        # Add progress callback
        def progress_callback(study, trial):
            if trial.number % 10 == 0:
                logger.info(
                    f"   Trial {trial.number}: Best value so far = {study.best_value:.4f}"
                )

        try:
            study.optimize(
                objective_wrapper,
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT,
                show_progress_bar=True,
                callbacks=[progress_callback],
            )
        except Exception as e:
            logger.error(f"Optuna optimization failed for {display_name}: {e}")
            # Continue with best trial found so far

        duration = time.time() - start_time

        logger.info(f"✅ Optuna for {display_name} completed in {duration:.1f}s")
        logger.info(
            f"   Best trial: {study.best_trial.number} with value: {study.best_trial.value:.4f}"
        )
        logger.info(f"   Total trials completed: {len(study.trials)}")

        # Train best model with full training data
        entry = _get_classifier_entry(display_name)
        _, module_path, _ = entry
        best_params = study.best_params

        # Ensure XGBoost uses CPU
        if "XGBoost" in display_name:
            best_params.setdefault("tree_method", "hist")
            best_params.setdefault("device", "cpu")

        best_model = get_model_instance(module_path, best_params)
        best_model.fit(X_train, y_train)

        # Evaluate on test set
        test_score = f1_score(y_test, best_model.predict(X_test), average="weighted")

        # Enhanced MLflow logging with study information
        try:
            study_info = {
                "n_trials": len(study.trials),
                "direction": "maximize",
                "best_trial": study.best_trial,
                "optimization_time": duration,
                "n_completed_trials": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                ),
                "n_failed_trials": len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                ),
            }

            self.tracker.log_tuning_run(
                model_name=display_name,
                best_model=best_model,
                best_params=best_params,
                best_score=test_score,
                X_test=X_test,
                y_test=y_test,
                tuning_method="optuna",
                study_info=study_info,
            )
        except Exception as e:
            logger.warning(f"MLflow logging failed for {display_name}: {e}")

        return best_model, best_params, test_score, study

    def save_tuned_model(self, model_name, model):
        """Save tuned model with additional metadata."""
        model_path = ARTIFACTS_DIR / f"{model_name}_tuned.joblib"

        # Save model with metadata
        model_data = {
            "model": model,
            "model_name": model_name,
            "tuning_method": "optuna",
            "saved_timestamp": time.time(),
        }

        joblib.dump(model_data, model_path)
        logger.info(f"💾 Saved tuned model to {model_path}")
        return model_path

    def tune_top_models(
        self,
        top_models,
        X_train,
        y_train,
        X_test,
        y_test,
        label_encoder=None,
        preprocessor=None,
    ):
        """
        Tune top models and evaluate them with comprehensive logging.

        Returns:
            tuple: (best_models_dict, evaluation_results)
        """
        logger.info(
            f"🎯 Starting hyperparameter tuning for {len(top_models)} models..."
        )

        best_models_dict = {}
        tuning_results = {}

        for i, display_name in enumerate(top_models, 1):
            logger.info(f"📊 Tuning model {i}/{len(top_models)}: {display_name}")

            try:
                # Tune model
                best_model, best_params, test_score, study = self.tune_with_optuna(
                    display_name, X_train, y_train, X_test, y_test
                )

                # Log results
                logger.info(f"✅ Tuned {display_name}:")
                logger.info(f"   Test F1-Score: {test_score:.4f}")
                logger.info(f"   Best params: {best_params}")

                best_models_dict[display_name] = {
                    "model": best_model,
                    "params": best_params,
                    "score": test_score,
                    "tuning_trials": len(study.trials),
                    "best_trial_number": study.best_trial.number,
                }

                tuning_results[display_name] = {
                    "study": study,
                    "best_score": test_score,
                    "tuning_time": study.best_trial.duration.total_seconds()
                    if study.best_trial.duration
                    else 0,
                }

                # Save model immediately
                model_path = self.save_tuned_model(display_name, best_model)
                best_models_dict[display_name]["saved_path"] = str(model_path)

                # Log intermediate success
                logger.info(f"🎉 {display_name} tuning completed successfully!")

            except Exception as e:
                logger.error(f"❌ Tuning failed for {display_name}: {e}")
                # Continue with other models
                continue

        if not best_models_dict:
            raise RuntimeError("All model tuning attempts failed!")

        # Log tuning summary
        logger.info(f"📋 Tuning Summary:")
        logger.info(f"   Models successfully tuned: {len(best_models_dict)}")
        for model_name, model_info in best_models_dict.items():
            logger.info(
                f"   {model_name}: F1={model_info['score']:.4f} (trials: {model_info['tuning_trials']})"
            )

        # Run comprehensive evaluations
        logger.info("📊 Starting comprehensive model evaluation...")
        try:
            from src.evaluate_models import evaluate_all_models

            evaluation_results = evaluate_all_models(
                best_models_dict, X_test, y_test, label_encoder, preprocessor
            )
            logger.info("✅ Model evaluation completed successfully")
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            # Return empty evaluation results but continue
            evaluation_results = {}

        return best_models_dict, evaluation_results


def get_best_model(
    top_models, X_train, y_train, X_test, y_test, label_encoder, preprocessor=None
):
    """Get the best model after tuning all top models"""
    tuner = ModelTuner()
    best_models_dict, _ = tuner.tune_top_models(
        top_models, X_train, y_train, X_test, y_test, label_encoder, preprocessor
    )

    # Find the model with the highest score
    if not best_models_dict:
        raise RuntimeError("No models were successfully tuned!")

    best_score = -1
    best_model = None
    best_model_name = None

    for model_name, model_info in best_models_dict.items():
        if model_info["score"] > best_score:
            best_score = model_info["score"]
            best_model = model_info["model"]
            best_model_name = model_name

    logger.info(
        f"🏆 Best model selected: {best_model_name} with F1-Score: {best_score:.4f}"
    )
    return best_model
