"""Enhanced project configuration with better path handling and model settings."""

import numpy as np
from pathlib import Path
import os
from typing import Dict

# Base paths - more flexible path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model artifacts directory with fallback paths
MODEL_ARTIFACTS_DIR = MODELS_DIR

# Alternative paths to search for model artifacts (in order of preference)
MODEL_ARTIFACT_SEARCH_PATHS = [
    MODELS_DIR,
    ARTIFACTS_DIR / "final_model",
    ARTIFACTS_DIR / "models",
    BASE_DIR / "model_artifacts",
    Path("./models"),
    Path("./artifacts/final_model"),
    Path("./"),
]

# Data configuration
TARGET_COL = "xaaladda_bukaanka"
DATA_PATH = DATA_DIR / "triage_data_cleaned.csv"

# Alternative data paths to search
DATA_SEARCH_PATHS = [
    DATA_DIR / "triage_data_cleaned.csv",
    DATA_DIR / "triage_data.csv",
    BASE_DIR / "triage_data_cleaned.csv",
    Path("./data/triage_data_cleaned.csv"),
    Path("./triage_data_cleaned.csv"),
]

# Model training configuration
RND = 42  # Random state for reproducibility
TEST_SIZE = 0.2
N_JOBS = 4
USE_SAMPLE = False
SAMPLE_SIZE = 5000
SAMPLE_RANDOM_STATE = RND

# Feature engineering
APPLY_SMOTE = False  # Disabled for better real-world performance

# Enhanced feature groups with validation
FEATURE_GROUPS = {
    "ordinal": [
        "heerka_qandhada",
        "muddada_qandhada",
        "madax_xanuun_daran",
        "muddada_madax_xanuunka",
        "muddada_qufaca",
        "muddada_xanuunka",
        "muddada_daalka",
        "muddada_mataga",
        "daal_badan",
        "matag_daran",
    ],
    "binary": [
        "qandho",
        "qufac",
        "madax_xanuun",
        "caloosh_xanuun",
        "daal",
        "matag",
        "dhaxan",
        "qufac_dhiig",
        "neeftu_dhibto",
        "iftiinka_dhibayo",
        "qoortu_adag_tahay",
        "lalabo",
        "shuban",
        "miisankaga_isdhimay",
        "qandho_daal_leh",
        "matag_dhiig_leh",
        "ceshan_karin_qoyaanka",
    ],
    "nominal": [
        "da'da",  # Note: API might use "da_da"
        "nooca_qufaca",
        "halka_xanuunku_kaa_hayo",
    ],
}

# Enhanced binary mappings with more variations
BINARY_MAP = {
    # Somali terms
    "haa": 1,
    "maya": 0,
    # English terms
    "yes": 1,
    "no": 0,
    # Numeric strings
    "1": 1,
    "0": 0,
    # Additional variations
    "true": 1,
    "false": 0,
    "y": 1,
    "n": 0,
    # Case variations
    "HAA": 1,
    "MAYA": 0,
    "YES": 1,
    "NO": 0,
    "True": 1,
    "False": 0,
}

# Enhanced ordinal level mappings
ORDINAL_LEVELS = {
    "mild": 0,
    "moderate": 1,
    "high": 2,
    # Alternative terms
    "low": 0,
    "medium": 1,
    "severe": 2,
    # Somali terms
    "yar": 0,
    "dhexdhexaad": 1,
    "daran": 2,
}

# Enhanced nominal category mappings
NOMINAL_MAPPINGS = {
    "da'da": {
        "young": 0,
        "middle_age": 1,
        "old": 2,
        # Alternative terms
        "dhalinyaro": 0,
        "dhex_da": 1,
        "waayeel": 2,
    },
    "nooca_qufaca": {
        "normal": 0,
        "dry": 1,
        "wet": 2,
        "bloody": 3,
        # Somali terms
        "caadi": 0,
        "qallalan": 1,
        "qoyan": 2,
        "dhiig_leh": 3,
    },
    "halka_xanuunku_kaa_hayo": {
        "chest": 0,
        "abdomen": 1,
        "head": 2,
        "back": 3,
        "limbs": 4,
        # Somali terms
        "laab": 0,
        "caloosh": 1,
        "madax": 2,
        "dhabar": 3,
        "lugaha": 4,
    },
}

# Logging configuration
LOG_FILE = LOGS_DIR / "medical_triage.log"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_MAX_BYTES = 10485760  # 10MB
LOG_BACKUP_COUNT = 5

# Enhanced Somali language mappings
SOMALI_MAPPINGS = {
    "risk_levels": {"low": "hoos", "medium": "dhexe", "high": "sare"},
    "confidence_levels": {"low": "hooseeya", "medium": "dhexdhexaad", "high": "sare"},
    "labels": {
        "heerka_halista": "Heerka Halista",
        "talooyin": "Talooyin",
        "tirada_calaamadaha": "Tirada Calaamadaha la Doortay",
        "kalsoonida": "Kalsoonida",
        "qiimeynta": "Qiimeynta",
    },
    "symptoms": {
        # Binary symptoms
        "qandho": "Qandho/Fever",
        "qufac": "Qufac/Cough",
        "madax_xanuun": "Madax Xanuun/Headache",
        "caloosh_xanuun": "Caloosh Xanuun/Stomach Pain",
        "daal": "Daal/Fatigue",
        "matag": "Matag/Nausea",
        # Add more mappings as needed
    },
}

# Model configuration with enhanced settings
CLASSIFIERS = [
    (
        "LogisticRegression",
        "sklearn.linear_model.LogisticRegression",
        {
            "solver": "liblinear",
            "max_iter": 2000,
            "random_state": RND,
            "class_weight": "balanced",  # Handle class imbalance
        },
    ),
    (
        "KNN",
        "sklearn.neighbors.KNeighborsClassifier",
        {"n_neighbors": 5, "weights": "distance"},
    ),
    (
        "DecisionTree",
        "sklearn.tree.DecisionTreeClassifier",
        {
            "random_state": RND,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
    ),
    (
        "RandomForest",
        "sklearn.ensemble.RandomForestClassifier",
        {
            "n_estimators": 100,
            "random_state": RND,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
    ),
    (
        "SVM",
        "sklearn.svm.LinearSVC",
        {
            "random_state": RND,
            "max_iter": 2000,
            "class_weight": "balanced",
            "dual": False,  # Recommended for n_samples > n_features
        },
    ),
    (
        "XGBoost",
        "xgboost.XGBClassifier",
        {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RND,
            "tree_method": "hist",
            "device": "cpu",
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
        },
    ),
    (
        "AdaBoost",
        "sklearn.ensemble.AdaBoostClassifier",
        {"n_estimators": 100, "learning_rate": 0.1, "random_state": RND},
    ),
    (
        "GradientBoosting",
        "sklearn.ensemble.GradientBoostingClassifier",
        {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "subsample": 0.8,
            "random_state": RND,
        },
    ),
    (
        "Bagging",
        "sklearn.ensemble.BaggingClassifier",
        {
            "n_estimators": 50,
            "max_samples": 0.8,
            "max_features": 0.8,
            "random_state": RND,
        },
    ),
]

# Enhanced hyperparameter tuning configurations
TUNING_CONFIGS = {
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 1.5, 2, 2.5],
    },
    "LogisticRegression": {
        "C": np.logspace(-3, 2, 20),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [500, 1000, 2000],
        "class_weight": ["balanced", None],
    },
    "KNN": {
        "n_neighbors": list(range(3, 31, 2)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski", "cosine"],
        "p": [1, 2],
    },
    "DecisionTree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20, 25],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": ["balanced", None],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 15, 20, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
    },
    "SVM": {
        "C": np.logspace(-2, 2, 15),
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 7)),
        "degree": [2, 3, 4],  # For poly kernel
        "class_weight": ["balanced", None],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "algorithm": ["SAMME", "SAMME.R"],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Bagging": {
        "n_estimators": [10, 50, 100, 200],
        "max_samples": [0.5, 0.7, 0.8, 1.0],
        "max_features": [0.5, 0.7, 0.8, 1.0],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
    },
}

# Cross-validation settings
CV_SPLITS = 5
TUNING_CV_SPLITS = 3

# Optuna hyperparameter tuning settings
OPTUNA_N_TRIALS = 150  # Increased for better optimization
OPTUNA_TIMEOUT = 3600  # 1 hour per model
OPTUNA_N_JOBS = 1  # Parallel jobs for Optuna
OPTUNA_SAMPLER = "TPE"  # Tree-structured Parzen Estimator

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "medical_triage_enhanced"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_ARTIFACT_LOCATION = str(ARTIFACTS_DIR / "mlruns")

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True
API_WORKERS = 1
API_TIMEOUT = 120  # seconds

# Prediction thresholds
CONFIDENCE_THRESHOLDS = {
    "high_confidence": 0.85,
    "medium_confidence": 0.65,
    "low_confidence": 0.45,
}

RISK_LEVEL_THRESHOLDS = {"high_risk": 0.8, "medium_risk": 0.6, "low_risk": 0.0}

# Validation settings
REQUIRED_MODEL_FILES = ["final_model.pkl", "preprocessor.pkl", "label_encoder.pkl"]

OPTIONAL_MODEL_FILES = [
    "feature_names.pkl",
    "feature_importances.pkl",
    "model_metadata.pkl",
]

# Dashboard configuration
DASHBOARD_CONFIG = {
    "page_title": "Nidaamka Qiimeynta Caafimaadka | Medical Triage System",
    "page_icon": "🏥",
    "layout": "wide",
    "theme": "light",
    "sidebar_state": "expanded",
    "max_prediction_history": 20,
    "auto_refresh_interval": 30,  # seconds
    "show_debug_info": False,
}

# Data validation rules
DATA_VALIDATION_RULES = {
    "ordinal_values": ["mild", "moderate", "high"],
    "binary_values": list(BINARY_MAP.keys()),
    "age_values": ["young", "middle_age", "old"],
    "cough_values": ["normal", "dry", "wet", "bloody"],
    "pain_location_values": ["chest", "abdomen", "head", "back", "limbs"],
    "required_features": (
        FEATURE_GROUPS["ordinal"] + FEATURE_GROUPS["binary"] + FEATURE_GROUPS["nominal"]
    ),
    "min_confidence_for_prediction": 0.3,
    "max_prediction_time_seconds": 30,
}

# Error messages
ERROR_MESSAGES = {
    "model_not_loaded": "Model artifacts not loaded. Please ensure training is completed.",
    "invalid_input": "Invalid input data. Please check feature values.",
    "prediction_failed": "Prediction failed. Please try again.",
    "low_confidence": "Prediction confidence is low. Please review inputs.",
    "api_unavailable": "API service is unavailable. Using local prediction.",
    "data_not_found": "Required data files not found.",
    "processing_timeout": "Request processing timed out.",
}

# Success messages
SUCCESS_MESSAGES = {
    "model_loaded": "Model artifacts loaded successfully.",
    "prediction_complete": "Prediction completed successfully.",
    "api_connected": "Successfully connected to API service.",
    "data_loaded": "Data loaded successfully.",
    "training_complete": "Model training completed successfully.",
}

# Feature importance settings
FEATURE_IMPORTANCE_CONFIG = {
    "max_features_to_show": 20,
    "importance_threshold": 0.01,
    "plot_style": "horizontal_bar",
    "color_scheme": "viridis",
}

# Evaluation metrics configuration
EVALUATION_METRICS = {
    "primary_metric": "f1_score",
    "secondary_metrics": ["accuracy", "precision", "recall"],
    "classification_metrics": ["confusion_matrix", "classification_report"],
    "probability_metrics": ["roc_auc", "average_precision"],
    "cross_validation_metric": "f1_weighted",
}

# Deployment configuration
DEPLOYMENT_CONFIG = {
    "model_version": "2.0.0",
    "api_version": "v2",
    "health_check_interval": 300,  # 5 minutes
    "model_warm_up": True,
    "cache_predictions": False,
    "enable_model_monitoring": True,
    "max_concurrent_requests": 100,
}


def get_model_artifact_path(filename: str) -> Path:
    """Get the path to a model artifact file, checking multiple locations."""
    for search_path in MODEL_ARTIFACT_SEARCH_PATHS:
        artifact_path = search_path / filename
        if artifact_path.exists():
            return artifact_path

    # Return default path if not found
    return MODEL_ARTIFACTS_DIR / filename


def get_data_path() -> Path:
    """Get the path to the data file, checking multiple locations."""
    for data_path in DATA_SEARCH_PATHS:
        if data_path.exists():
            return data_path

    # Return default path if not found
    return DATA_PATH


def validate_environment() -> Dict[str, bool]:
    """Validate the environment setup."""
    validation_results = {
        "directories_exist": all(
            path.exists() for path in [MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR, LOGS_DIR]
        ),
        "data_accessible": get_data_path().exists(),
        "model_artifacts_accessible": any(
            (path / "final_model.pkl").exists()
            or (path / "best_model.pkl").exists()
            or list(path.glob("*_production.pkl"))
            or list(path.glob("*_tuned.joblib"))
            for path in MODEL_ARTIFACT_SEARCH_PATHS
            if path.exists()
        ),
        "config_valid": all(
            [
                len(FEATURE_GROUPS["ordinal"]) > 0,
                len(FEATURE_GROUPS["binary"]) > 0,
                len(FEATURE_GROUPS["nominal"]) > 0,
                len(CLASSIFIERS) > 0,
                OPTUNA_N_TRIALS > 0,
            ]
        ),
    }

    return validation_results


# Export key paths for easy access
__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "ARTIFACTS_DIR",
    "PLOTS_DIR",
    "LOGS_DIR",
    "MODEL_ARTIFACTS_DIR",
    "MODEL_ARTIFACT_SEARCH_PATHS",
    "DATA_SEARCH_PATHS",
    "TARGET_COL",
    "DATA_PATH",
    "FEATURE_GROUPS",
    "BINARY_MAP",
    "SOMALI_MAPPINGS",
    "CLASSIFIERS",
    "TUNING_CONFIGS",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_TRACKING_URI",
    "get_model_artifact_path",
    "get_data_path",
    "validate_environment",
]
