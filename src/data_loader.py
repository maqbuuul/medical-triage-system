import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.config import (
    DATA_PATH,
    TARGET_COL,
    USE_SAMPLE,
    SAMPLE_SIZE,
    SAMPLE_RANDOM_STATE,
    BINARY_MAP,
    FEATURE_GROUPS,
)


def load_data(
    path=DATA_PATH,
    use_sample: bool = USE_SAMPLE,
    sample_size: int = SAMPLE_SIZE,
    sample_random_state: int = SAMPLE_RANDOM_STATE,
    target_col: str = TARGET_COL,
):
    df = pd.read_csv(path)

    if use_sample and len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=sample_random_state)
    else:
        df_sampled = df.copy()

    # Preprocess binary columns
    for col in FEATURE_GROUPS.get("binary", []):
        if col in df_sampled.columns:
            df_sampled[col] = df_sampled[col].map(BINARY_MAP).fillna(df_sampled[col])

    if target_col not in df_sampled.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    X = df_sampled.drop(columns=[target_col])
    y = df_sampled[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    return X, y_encoded, le, df_sampled
