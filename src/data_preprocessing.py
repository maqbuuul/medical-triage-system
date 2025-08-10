"""Enhanced data preprocessing module."""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.logging_config import get_logger
from src.config import FEATURE_GROUPS, BINARY_MAP, RND, TEST_SIZE, APPLY_SMOTE
from src.data_loader import load_data as load_sampled_data

logger = get_logger("data_preprocessing")


def replace_str_nan(X):
    return np.where((X == "0") | (X == "nan") | (X == ""), np.nan, X)


def create_preprocessor():
    logger.info("Creating preprocessing pipeline...")

    ord_pipe = Pipeline(
        [
            ("to_nan", FunctionTransformer(replace_str_nan)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[["mild", "moderate", "high"]]
                    * len(FEATURE_GROUPS["ordinal"]),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    bin_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    nom_pipe = Pipeline(
        [
            ("to_nan", FunctionTransformer(replace_str_nan)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("ord", ord_pipe, FEATURE_GROUPS["ordinal"]),
            ("bin", bin_pipe, FEATURE_GROUPS["binary"]),
            ("nom", nom_pipe, FEATURE_GROUPS["nominal"]),
        ],
        remainder="drop",
    )

    logger.info("✅ Preprocessing pipeline created")
    return preprocessor


def print_distribution(arr, le, name):
    cnts = Counter(arr)
    total = len(arr)
    logger.info(f"\n{name} distribution:")
    logger.info("-" * 40)
    for lbl, ct in cnts.items():
        class_name = le.inverse_transform([lbl])[0] if le else str(lbl)
        percentage = ct / total * 100
        logger.info(f"  {class_name:<20} {ct:>4} ({percentage:>5.1f}%)")
    logger.info("-" * 40)


def apply_smote(X, y, label_encoder):
    logger.info("Applying SMOTE oversampling...")
    class_counts = Counter(y)
    k_neighbors = min(5, min(class_counts.values()) - 1)
    smote = SMOTE(random_state=RND, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(f"✅ SMOTE applied: Original: {X.shape} → Resampled: {X_res.shape}")
    return X_res, y_res


def preprocess_pipeline():
    logger.info("Starting data preprocessing pipeline...")
    try:
        X_df, y_enc, le, df_sampled = load_sampled_data()
        logger.info(
            f"✅ Data loaded: {X_df.shape[0]} samples, {X_df.shape[1]} features"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_enc, test_size=TEST_SIZE, random_state=RND, stratify=y_enc
        )
        logger.info(
            f"✅ Data split: Train: {X_train.shape[0]}, Test: {X_test.shape[0]}"
        )

        preprocessor = create_preprocessor()
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        logger.info(
            f"✅ Data transformed: Train: {X_train_trans.shape}, Test: {X_test_trans.shape}"
        )

        if APPLY_SMOTE:
            X_train_final, y_train_final = apply_smote(X_train_trans, y_train, le)
        else:
            X_train_final, y_train_final = X_train_trans, y_train

        return {
            "X_train": X_train_final,
            "y_train": y_train_final,
            "X_test": X_test_trans,
            "y_test": y_test,
            "preprocessor": preprocessor,
            "label_encoder": le,
            "sampled_df": df_sampled,
        }
    except Exception as e:
        logger.error(f"❌ Data preprocessing failed: {e}")
        raise
