# predict.py
"""
Prediction script for medical triage pipeline using the final trained model.
If the model is not found, trains it first.
"""

import os
import sys
import joblib
import pandas as pd
from pathlib import Path

from src.config import MODELS_DIR, TARGET_COL
from logging_config import get_logger
from src.utils.data_utils import preprocess_new_data

logger = get_logger("predict")

MODEL_PATH = Path(MODELS_DIR) / "final_model.pkl"
ENCODER_PATH = Path(MODELS_DIR) / "label_encoder.pkl"


def load_model_and_encoder():
    """Load trained model and label encoder, or train if missing."""
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        logger.warning("No trained model found. Running training first...")
        os.system(f"{sys.executable} train_models.py")  # Call training script

    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


def preprocess_new_data(df: pd.DataFrame):
    """Apply preprocessing steps to match training pipeline."""
    from data_preprocessing import preprocess_new_data_pipeline

    return preprocess_new_data_pipeline(df)


def predict_on_new_data(input_path: str, output_path: str):
    """Predict triage levels for new dataset."""
    # Load model + encoder
    model, label_encoder = load_model_and_encoder()

    # Load new data
    logger.info(f"Reading new dataset from {input_path}")
    df = pd.read_csv(input_path)

    # Preprocess
    X_processed = preprocess_new_data(df)

    # Predict
    preds = model.predict(X_processed)
    preds_labels = label_encoder.inverse_transform(preds)

    # Save predictions
    df["predicted_" + TARGET_COL] = preds_labels
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <new_dataset.csv> <output_predictions.csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    predict_on_new_data(input_csv, output_csv)
