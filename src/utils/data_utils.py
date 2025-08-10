"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from src.config import FEATURE_GROUPS, BINARY_MAP, RND
from src.logging_config import get_logger

logger = get_logger("data_utils")


def replace_str_nan(X):
    """Replace string representations of NaN."""
    return np.where((X == "0") | (X == "nan") | (X == ""), np.nan, X)


def preprocess_features(df):
    """Preprocess features before transformation."""
    df = df.copy()

    # Convert feature columns to string
    for col in sum(FEATURE_GROUPS.values(), []):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Apply binary mapping
    for col in FEATURE_GROUPS["binary"]:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP).fillna(df[col])

    return df


def create_feature_names(preprocessor):
    """Create feature names after preprocessing."""
    feature_names = []
    feature_names.extend(FEATURE_GROUPS["ordinal"])
    feature_names.extend(FEATURE_GROUPS["binary"])

    # Handle nominal features
    try:
        nominal_transformer = preprocessor.named_transformers_["nom"]
        onehot_encoder = nominal_transformer.named_steps["onehot"]
        nominal_features = onehot_encoder.get_feature_names_out(
            FEATURE_GROUPS["nominal"]
        )
        feature_names.extend(nominal_features)
    except Exception as e:
        logger.warning(f"Couldn't extract one-hot features: {e}")
        for feature in FEATURE_GROUPS["nominal"]:
            feature_names.append(f"{feature}_encoded")

    return feature_names
