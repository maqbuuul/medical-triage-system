"""
FastAPI service for medical triage model predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, field_validator
import logging
from contextlib import asynccontextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from src.config import MODEL_ARTIFACTS_DIR, FEATURE_GROUPS, BINARY_MAP, SOMALI_MAPPINGS
from src.logging_config import setup_logging

# Setup logging
logger = setup_logging("INFO", "logs/fastapi.log")

# Global variables for model artifacts
model = None
preprocessor = None
label_encoder = None
feature_names = None
sample_data = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for predictions."""

    # Ordinal features
    heerka_qandhada: Optional[str] = "mild"
    muddada_qandhada: Optional[str] = "mild"
    madax_xanuun_daran: Optional[str] = "mild"
    muddada_madax_xanuunka: Optional[str] = "mild"
    muddada_qufaca: Optional[str] = "mild"
    muddada_xanuunka: Optional[str] = "mild"
    muddada_daalka: Optional[str] = "mild"
    muddada_mataga: Optional[str] = "mild"
    daal_badan: Optional[str] = "mild"
    matag_daran: Optional[str] = "mild"

    # Binary features
    qandho: Optional[str] = "maya"
    qufac: Optional[str] = "maya"
    madax_xanuun: Optional[str] = "maya"
    caloosh_xanuun: Optional[str] = "maya"
    daal: Optional[str] = "maya"
    matag: Optional[str] = "maya"
    dhaxan: Optional[str] = "maya"
    qufac_dhiig: Optional[str] = "maya"
    neeftu_dhibto: Optional[str] = "maya"
    iftiinka_dhibayo: Optional[str] = "maya"
    qoortu_adag_tahay: Optional[str] = "maya"
    lalabo: Optional[str] = "maya"
    shuban: Optional[str] = "maya"
    miisankaga_isdhimay: Optional[str] = "maya"
    qandho_daal_leh: Optional[str] = "maya"
    matag_dhiig_leh: Optional[str] = "maya"
    ceshan_karin_qoyaanka: Optional[str] = "maya"

    # Nominal features
    da_da: Optional[str] = "middle_age"
    nooca_qufaca: Optional[str] = "normal"
    halka_xanuunku_kaa_hayo: Optional[str] = "chest"

    @field_validator("*", mode="before")
    def validate_features(cls, v, info):
        field_name = info.field_name

        if field_name in FEATURE_GROUPS["ordinal"]:
            if v not in ["mild", "moderate", "high"]:
                return "mild"  # Default value
        elif field_name in FEATURE_GROUPS["binary"]:
            if v not in ["haa", "maya", "yes", "no", "1", "0"]:
                return "maya"  # Default value

        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_label: str
    predicted_class: str
    confidence: float
    risk_level: str
    risk_level_somali: str
    num_symptoms: int
    matching_tips: List[str]
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


def load_model_artifacts():
    """Load all model artifacts on startup."""
    global model, preprocessor, label_encoder, feature_names, sample_data

    try:
        # Use the new config variable for model files path
        model_path = MODEL_ARTIFACTS_DIR / "final_model.pkl"
        preprocessor_path = MODEL_ARTIFACTS_DIR / "preprocessor.pkl"
        encoder_path = MODEL_ARTIFACTS_DIR / "label_encoder.pkl"

        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Loaded model from: {model_path}")

        # Load preprocessor
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded preprocessor from: {preprocessor_path}")

        # Load label encoder
        if not encoder_path.exists():
            raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
        label_encoder = joblib.load(encoder_path)
        logger.info(f"Loaded label encoder from: {encoder_path}")

        # Optional: load feature names if you have that file, else skip
        feature_names_path = MODEL_ARTIFACTS_DIR / "feature_names.pkl"
        if feature_names_path.exists():
            feature_names = joblib.load(feature_names_path)
            logger.info(f"Loaded feature names from: {feature_names_path}")

        # Load sample data for tips matching
        data_path = Path("data/triage_data_cleaned.csv")
        if data_path.exists():
            sample_data = pd.read_csv(data_path)
            logger.info(f"Loaded sample data for tips matching")

        logger.info("All model artifacts loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return False


def preprocess_input(request_data: dict) -> np.ndarray:
    """Preprocess input data for prediction."""
    try:
        # Create DataFrame with the same structure as training data
        input_df = pd.DataFrame([request_data])

        # Handle feature name mapping
        if "da_da" in input_df.columns:
            input_df = input_df.rename(columns={"da_da": "da'da"})

        # Process binary features
        for col in FEATURE_GROUPS["binary"]:
            if col in input_df.columns:
                input_df[col] = input_df[col].map(BINARY_MAP).fillna(0)

        # Ensure all required columns exist
        all_features = (
            FEATURE_GROUPS["ordinal"]
            + FEATURE_GROUPS["binary"]
            + FEATURE_GROUPS["nominal"]
        )

        for feature in all_features:
            if feature not in input_df.columns:
                if feature in FEATURE_GROUPS["binary"]:
                    input_df[feature] = 0
                elif feature in FEATURE_GROUPS["ordinal"]:
                    input_df[feature] = "mild"
                else:
                    input_df[feature] = "unknown"

        # Reorder columns to match training data
        input_df = input_df[all_features]

        # Apply preprocessing
        processed_data = preprocessor.transform(input_df)

        return processed_data

    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise


def count_symptoms(request_data: dict) -> int:
    """Count number of positive symptoms."""
    symptom_count = 0

    for feature, value in request_data.items():
        if feature in FEATURE_GROUPS["binary"]:
            if str(value).lower() in ["haa", "yes", "1"]:
                symptom_count += 1
        elif feature in FEATURE_GROUPS["ordinal"]:
            if str(value).lower() in ["moderate", "high"]:
                symptom_count += 1

    return symptom_count


def get_matching_tips(predicted_class: str, num_symptoms: int) -> List[str]:
    """Get matching tips from the dataset."""
    try:
        if sample_data is None:
            return ["Taloo guud: Fadlan la tashii dhakhtarka."]

        # Filter data based on predicted class and similar symptom count
        matching_data = sample_data[
            (sample_data["xaaladda_bukaanka"] == predicted_class)
        ]

        if matching_data.empty:
            return ["Taloo guud: Fadlan la tashii dhakhtarka."]

        # Get tips from matching cases
        tips = []
        if "talooyin" in matching_data.columns:
            sample_tips = matching_data["talooyin"].dropna().unique()
            tips.extend([tip for tip in sample_tips if len(str(tip)) > 5][:3])

        # Default tips if none found
        if not tips:
            tips = [
                "Fadlan la tashii dhakhtarka si loo helo baaris dheeri ah.",
                "Raadi caawimaad caafimaad hadii calaamadahu sii daraan.",
                "Raacaba halka xanuunku u horumaro oo qor wax kasta oo muhiim ah.",
            ]

        return tips[:3]  # Return max 3 tips

    except Exception as e:
        logger.error(f"Error getting matching tips: {e}")
        return ["Taloo guud: Fadlan la tashii dhakhtarka."]


@asynccontextmanager
async def lifespan(app: FastAPI):
    success = load_model_artifacts()
    if not success:
        logger.error("Failed to load model artifacts on startup")
    yield
    # Shutdown code here if needed


app = FastAPI(
    title="Medical Triage API",
    description="API for medical triage risk prediction using ML",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy", model_loaded=model is not None, version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, http_request: Request):
    """Make prediction for given symptoms."""

    # Log request
    client_ip = http_request.client.host
    logger.info(f"Prediction request from {client_ip}")

    try:
        if model is None or preprocessor is None or label_encoder is None:
            raise HTTPException(
                status_code=503, detail="Model not loaded. Please try again later."
            )

        # Convert request to dict
        request_data = request.dict()

        # Preprocess input
        processed_input = preprocess_input(request_data)

        # Make prediction
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]

        # Get class names and probabilities
        class_names = label_encoder.classes_
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(prediction_proba))

        # Create probability dictionary
        all_probabilities = {
            str(class_name): float(prob)
            for class_name, prob in zip(class_names, prediction_proba)
        }

        # Determine risk level
        risk_level = (
            "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        )
        risk_level_somali = SOMALI_MAPPINGS["risk_levels"].get(risk_level, risk_level)

        # Count symptoms
        num_symptoms = count_symptoms(request_data)

        # Get matching tips
        matching_tips = get_matching_tips(predicted_class, num_symptoms)

        # Create response
        response = PredictionResponse(
            predicted_label=str(prediction),
            predicted_class=predicted_class,
            confidence=confidence,
            risk_level=risk_level,
            risk_level_somali=risk_level_somali,
            num_symptoms=num_symptoms,
            matching_tips=matching_tips,
            all_probabilities=all_probabilities,
        )

        # Log successful prediction
        logger.info(
            f"Prediction successful: {predicted_class} (confidence: {confidence:.3f})"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get model information."""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        info = {
            "model_type": type(model).__name__,
            "features_count": len(feature_names) if feature_names else "unknown",
            "classes": list(label_encoder.classes_) if label_encoder else [],
            "feature_groups": FEATURE_GROUPS,
        }

        return info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
