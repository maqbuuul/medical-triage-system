"""Project configuration and hyperparameter search spaces."""

import numpy as np
from pathlib import Path

# Reproducibility
RND = 42
TEST_SIZE = 0.2
TARGET_COL = "xaaladda_bukaanka"
DATA_PATH = Path("./data/triage_data_cleaned.csv")

# Parallelism
N_JOBS = 4

# Sampling
USE_SAMPLE = False
SAMPLE_SIZE = 5000
SAMPLE_RANDOM_STATE = RND

# Disable SMOTE for better real-world performance
APPLY_SMOTE = False

# Feature groups
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
    "nominal": ["da'da", "nooca_qufaca", "halka_xanuunku_kaa_hayo"],
}

BINARY_MAP = {"haa": 1, "yes": 1, "1": 1, "maya": 0, "no": 0, "0": 0}

# Directories
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
LOG_FILE = Path("logs/medical_triage.log")
MODEL_ARTIFACTS_DIR = Path("models")

# Somali language mappings
SOMALI_MAPPINGS = {
    "risk_levels": {"low": "hoos", "medium": "dhexe", "high": "sare"},
    "labels": {
        "heerka_halista": "Heerka Halista",
        "talooyin": "Talooyin",
        "tirada_calaamadaha": "Tirada Calaamadaha la Doortay",
    },
}

# Model configuration
CLASSIFIERS = [
    (
        "LogisticRegression",
        "sklearn.linear_model.LogisticRegression",
        {"solver": "liblinear", "max_iter": 2000, "random_state": RND},
    ),
    ("KNN", "sklearn.neighbors.KNeighborsClassifier", {}),
    ("DecisionTree", "sklearn.tree.DecisionTreeClassifier", {"random_state": RND}),
    (
        "RandomForest",
        "sklearn.ensemble.RandomForestClassifier",
        {"n_estimators": 100, "random_state": RND},
    ),
    ("SVM", "sklearn.svm.LinearSVC", {"random_state": RND, "max_iter": 2000}),
    (
        "XGBoost",
        "xgboost.XGBClassifier",
        {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "tree_method": "hist",  # Fixed for CPU compatibility
            "device": "cpu",  # Explicitly set to CPU
        },
    ),
    ("AdaBoost", "sklearn.ensemble.AdaBoostClassifier", {"random_state": RND}),
    (
        "GradientBoosting",
        "sklearn.ensemble.GradientBoostingClassifier",
        {"random_state": RND},
    ),
    ("Bagging", "sklearn.ensemble.BaggingClassifier", {"random_state": RND}),
]

# Hyperparameter tuning configurations
TUNING_CONFIGS = {
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 1.5, 2],
        "scale_pos_weight": [1],
    },
    "LogisticRegression": {
        "C": np.logspace(-3, 2, 20),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [100, 500, 1000, 2000],
    },
    "KNN": {
        "n_neighbors": list(range(1, 31)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2],
    },
    "DecisionTree": {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    },
    "SVM": {
        "C": np.logspace(-2, 2, 10),
        "gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 5)),
        "kernel": ["linear", "rbf"],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
    "Bagging": {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.7, 1.0],
        "max_features": [0.5, 0.7, 1.0],
        "bootstrap": [True, False],
    },
}

# Cross-validation
CV_SPLITS = 5
TUNING_CV_SPLITS = 3

# Increase tuning trials
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 hour per model

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "medical_triage_pipeline"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
