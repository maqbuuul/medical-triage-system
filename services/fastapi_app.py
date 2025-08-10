"""
Enhanced FastAPI service for medical triage model predictions with improved error handling and model loading.
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
import uvicorn

# Add current directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from src.config import MODEL_ARTIFACTS_DIR, FEATURE_GROUPS, BINARY_MAP, SOMALI_MAPPINGS
from src.logging_config import setup_logging

# Setup enhanced logging
logger = setup_logging("INFO", "logs/fastapi.log")

# Global variables for model artifacts
model = None
preprocessor = None
label_encoder = None
feature_names = None
sample_data = None
model_metadata = {}


# Enhanced Pydantic models
class PredictionRequest(BaseModel):
    """Enhanced request model for predictions with validation."""

    # Ordinal features with defaults
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

    # Binary features with defaults
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

    # Nominal features with defaults
    da_da: Optional[str] = "middle_age"  # Note: API uses da_da, will map to da'da
    nooca_qufaca: Optional[str] = "normal"
    halka_xanuunku_kaa_hayo: Optional[str] = "chest"

    @field_validator("*", mode="before")
    def validate_features(cls, v, info):
        """Enhanced field validation with logging."""
        field_name = info.field_name

        if field_name in FEATURE_GROUPS["ordinal"]:
            if v not in ["mild", "moderate", "high"]:
                logger.warning(
                    f"Invalid ordinal value for {field_name}: {v}, using 'mild'"
                )
                return "mild"
        elif field_name in FEATURE_GROUPS["binary"]:
            if v not in ["haa", "maya", "yes", "no", "1", "0"]:
                logger.warning(
                    f"Invalid binary value for {field_name}: {v}, using 'maya'"
                )
                return "maya"
        elif field_name in ["da_da"]:  # Age field
            if v not in ["young", "middle_age", "old"]:
                logger.warning(f"Invalid age value: {v}, using 'middle_age'")
                return "middle_age"
        elif field_name in ["nooca_qufaca"]:  # Cough type
            if v not in ["normal", "dry", "wet", "bloody"]:
                logger.warning(f"Invalid cough type: {v}, using 'normal'")
                return "normal"
        elif field_name in ["halka_xanuunku_kaa_hayo"]:  # Pain location
            if v not in ["chest", "abdomen", "head", "back", "limbs"]:
                logger.warning(f"Invalid pain location: {v}, using 'chest'")
                return "chest"

        return v


class PredictionResponse(BaseModel):
    """Enhanced response model for predictions."""

    predicted_label: str
    predicted_class: str
    confidence: float
    risk_level: str
    risk_level_somali: str
    num_symptoms: int
    matching_tips: List[str]
    all_probabilities: Dict[str, float]
    model_info: Dict[str, str]
    prediction_metadata: Dict[str, any]


class HealthResponse(BaseModel):
    """Enhanced health check response."""

    status: str
    model_loaded: bool
    version: str
    api_status: str
    artifacts_status: Dict[str, bool]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Detailed model information response."""

    model_type: str
    features_count: int
    classes: List[str]
    feature_groups: Dict[str, List[str]]
    model_metadata: Dict[str, any]
    artifacts_loaded: Dict[str, bool]


def load_model_artifacts():
    """Enhanced model artifact loading with multiple fallback paths."""
    global \
        model, \
        preprocessor, \
        label_encoder, \
        feature_names, \
        sample_data, \
        model_metadata

    logger.info("🔄 Loading model artifacts...")

    # Multiple possible paths for model artifacts
    possible_model_dirs = [
        MODEL_ARTIFACTS_DIR,
        Path("models"),
        Path("../models"),
        Path("./models"),
        Path("artifacts/final_model"),
        Path("../artifacts/final_model"),
    ]

    model_files = {
        "model": [
            "final_model.pkl",
            "best_model.pkl",
            "*_production.pkl",
            "*_tuned.joblib",
        ],
        "preprocessor": ["preprocessor.pkl"],
        "label_encoder": ["label_encoder.pkl"],
        "feature_names": ["feature_names.pkl"],
    }

    loaded_artifacts = {}

    for model_dir in possible_model_dirs:
        if not model_dir.exists():
            logger.debug(f"Directory not found: {model_dir}")
            continue

        logger.info(f"🔍 Checking directory: {model_dir}")

        for artifact_name, file_patterns in model_files.items():
            if artifact_name in loaded_artifacts:
                continue  # Already loaded

            for pattern in file_patterns:
                try:
                    if "*" in pattern:
                        matching_files = list(model_dir.glob(pattern))
                        if matching_files:
                            artifact_path = matching_files[0]
                        else:
                            continue
                    else:
                        artifact_path = model_dir / pattern

                    if artifact_path.exists():
                        logger.info(f"📂 Loading {artifact_name} from: {artifact_path}")

                        # Load the artifact
                        if artifact_name == "model":
                            # Handle different model file formats
                            if str(artifact_path).endswith(".joblib"):
                                loaded_data = joblib.load(artifact_path)
                                if (
                                    isinstance(loaded_data, dict)
                                    and "model" in loaded_data
                                ):
                                    loaded_artifacts[artifact_name] = loaded_data[
                                        "model"
                                    ]
                                    model_metadata.update(loaded_data)
                                else:
                                    loaded_artifacts[artifact_name] = loaded_data
                            else:
                                loaded_artifacts[artifact_name] = joblib.load(
                                    artifact_path
                                )
                        else:
                            loaded_artifacts[artifact_name] = joblib.load(artifact_path)

                        logger.info(f"✅ Successfully loaded {artifact_name}")
                        break

                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to load {artifact_name} from {artifact_path}: {e}"
                    )
                    continue

    # Assign to global variables
    model = loaded_artifacts.get("model")
    preprocessor = loaded_artifacts.get("preprocessor")
    label_encoder = loaded_artifacts.get("label_encoder")
    feature_names = loaded_artifacts.get("feature_names")

    # Load sample data for tips
    try:
        possible_data_paths = [
            Path("data/triage_data_cleaned.csv"),
            Path("../data/triage_data_cleaned.csv"),
            Path("./data/triage_data_cleaned.csv"),
            Path("triage_data_cleaned.csv"),
        ]

        for data_path in possible_data_paths:
            if data_path.exists():
                sample_data = pd.read_csv(data_path)
                logger.info(f"📊 Loaded sample data from: {data_path}")
                break
        else:
            logger.warning("⚠️ Sample data not found")

    except Exception as e:
        logger.warning(f"⚠️ Failed to load sample data: {e}")

    # Validation
    required_artifacts = ["model", "preprocessor", "label_encoder"]
    loaded_required = [name for name in required_artifacts if name in loaded_artifacts]
    missing_required = [
        name for name in required_artifacts if name not in loaded_artifacts
    ]

    if loaded_required:
        logger.info(f"✅ Successfully loaded required artifacts: {loaded_required}")

    if missing_required:
        logger.error(f"❌ Missing critical artifacts: {missing_required}")
        logger.error(
            "🔧 Please ensure model training has completed and artifacts are saved properly"
        )
        return False

    logger.info("🎉 All model artifacts loaded successfully")
    return True


def enhanced_preprocess_input(request_data: dict) -> np.ndarray:
    """Enhanced input preprocessing with better error handling."""
    try:
        logger.info(f"🔄 Preprocessing input data: {len(request_data)} features")

        # Create DataFrame
        input_df = pd.DataFrame([request_data])

        # Handle feature name mapping (da_da -> da'da)
        if "da_da" in input_df.columns:
            input_df = input_df.rename(columns={"da_da": "da'da"})
            logger.debug("✅ Mapped da_da to da'da")

        # Process binary features with enhanced mapping
        for col in FEATURE_GROUPS["binary"]:
            if col in input_df.columns:
                original_value = input_df[col].iloc[0]
                mapped_value = BINARY_MAP.get(str(original_value).lower(), 0)
                input_df[col] = mapped_value
                logger.debug(
                    f"Binary feature {col}: {original_value} -> {mapped_value}"
                )

        # Ensure all required columns exist with appropriate defaults
        all_features = (
            FEATURE_GROUPS["ordinal"]
            + FEATURE_GROUPS["binary"]
            + FEATURE_GROUPS["nominal"]
        )

        # Add missing features with defaults
        for feature in all_features:
            if feature not in input_df.columns:
                if feature in FEATURE_GROUPS["binary"]:
                    default_value = 0
                elif feature in FEATURE_GROUPS["ordinal"]:
                    default_value = "mild"
                elif feature in FEATURE_GROUPS["nominal"]:
                    if feature == "da'da":
                        default_value = "middle_age"
                    elif feature == "nooca_qufaca":
                        default_value = "normal"
                    elif feature == "halka_xanuunku_kaa_hayo":
                        default_value = "chest"
                    else:
                        default_value = "unknown"
                else:
                    default_value = "unknown"

                input_df[feature] = default_value
                logger.debug(
                    f"Added missing feature {feature} with default: {default_value}"
                )

        # Reorder columns to match training data
        input_df = input_df[all_features]
        logger.debug(f"✅ Features reordered: {input_df.columns.tolist()}")

        # Apply preprocessing
        processed_data = preprocessor.transform(input_df)
        logger.info(f"✅ Preprocessing completed: shape {processed_data.shape}")

        return processed_data

    except Exception as e:
        logger.error(f"❌ Error preprocessing input: {e}")
        logger.error(f"Input data: {request_data}")
        raise HTTPException(
            status_code=400, detail=f"Input preprocessing failed: {str(e)}"
        )


def enhanced_count_symptoms(request_data: dict) -> Dict[str, int]:
    """Enhanced symptom counting with detailed breakdown."""
    symptom_breakdown = {
        "total": 0,
        "binary_positive": 0,
        "severity_moderate": 0,
        "severity_high": 0,
        "weighted_score": 0,
    }

    for feature, value in request_data.items():
        if feature in FEATURE_GROUPS["binary"]:
            if str(value).lower() in ["haa", "yes", "1"]:
                symptom_breakdown["binary_positive"] += 1
                symptom_breakdown["total"] += 1
                symptom_breakdown["weighted_score"] += 1

        elif feature in FEATURE_GROUPS["ordinal"]:
            if str(value).lower() == "moderate":
                symptom_breakdown["severity_moderate"] += 1
                symptom_breakdown["total"] += 1
                symptom_breakdown["weighted_score"] += 1.5
            elif str(value).lower() == "high":
                symptom_breakdown["severity_high"] += 1
                symptom_breakdown["total"] += 1
                symptom_breakdown["weighted_score"] += 2

    return symptom_breakdown


def get_enhanced_matching_tips(
    predicted_class: str, symptom_breakdown: Dict, confidence: float
) -> List[str]:
    """Get enhanced matching tips based on prediction details."""
    try:
        tips = []

        # Risk-based tips
        if confidence > 0.85:
            if (
                "urgent" in predicted_class.lower()
                or "emergency" in predicted_class.lower()
            ):
                tips.extend(
                    [
                        "🚨 Deg deg u tag cisbitaalka ama wac lambarka degdegga ah",
                        "⚡ Tani waa xaalad deg deg ah oo u baahan caawimaad dhakhtar",
                        "🏥 Ha iska dhaafin - raadi caawimaad isla markiiba",
                    ]
                )
            elif symptom_breakdown["total"] > 8:
                tips.extend(
                    [
                        "🩺 La tashii dhakhtar 24 saacadood gudahood",
                        "📋 Qor dhammaan calaamadahaaga oo la tag dhakhtarka",
                        "⚠️ Kormeer calaamadahaaga si joogto ah",
                    ]
                )
            else:
                tips.extend(
                    [
                        "🩺 La tashii dhakhtar dhowaan si loo hubo",
                        "📝 Raacaba calaamadahaaga muddo dhawr maalmood ah",
                        "💊 Isticmaal dawooyin lagu yaraynayo xanuunka",
                    ]
                )
        else:
            tips.extend(
                [
                    "🏠 Naso guriga oo cab biyo badan",
                    "🍵 Cab shaah kulul iyo caano",
                    "📞 Wac dhakhtar haddii calaamadahu sii daraan",
                ]
            )

        # Add severity-specific tips
        if symptom_breakdown["severity_high"] > 2:
            tips.append(
                "🔴 Calaamado aad u daran ayaa jira - deg deg dhakhtar la tashii"
            )
        elif symptom_breakdown["severity_moderate"] > 3:
            tips.append("🟡 Calaamado dhexdhexaad ah - kormeer dhow u samee")

        # Add condition-specific tips from sample data
        if sample_data is not None:
            try:
                matching_data = sample_data[
                    sample_data["xaaladda_bukaanka"] == predicted_class
                ]
                if not matching_data.empty and "talooyin" in matching_data.columns:
                    sample_tips = matching_data["talooyin"].dropna().unique()
                    additional_tips = [
                        tip for tip in sample_tips if len(str(tip)) > 10
                    ][:2]
                    tips.extend(additional_tips)
            except Exception as e:
                logger.warning(f"Could not get sample-based tips: {e}")

        # Remove duplicates and limit
        unique_tips = []
        for tip in tips:
            if tip not in unique_tips:
                unique_tips.append(tip)

        return unique_tips[:5]  # Return max 5 tips

    except Exception as e:
        logger.error(f"Error generating tips: {e}")
        return [
            "🩺 Fadlan la tashii dhakhtarka si loo helo baaris dheeri ah",
            "📞 Wac xafiiska caafimaadka haddii calaamadahu sii daraan",
            "🏥 Tag cisbitaalka haddii xaaladdu si degdeg ah u darto",
        ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management."""
    logger.info("🚀 Starting Medical Triage API...")

    success = load_model_artifacts()
    if success:
        logger.info("✅ API startup completed successfully")
    else:
        logger.error("❌ API startup failed - some features may not work")

    yield

    logger.info("🛑 Shutting down Medical Triage API...")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Medical Triage AI API",
    description="Enhanced API for medical triage risk prediction using machine learning",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enhanced middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = pd.Timestamp.now()

    # Log request
    logger.info(f"📥 {request.method} {request.url.path} from {request.client.host}")

    response = await call_next(request)

    # Log response
    process_time = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"📤 Response: {response.status_code} in {process_time:.3f}s")

    return response


@app.get("/", response_model=HealthResponse)
async def root():
    """Enhanced root endpoint with detailed status."""
    artifacts_status = {
        "model": model is not None,
        "preprocessor": preprocessor is not None,
        "label_encoder": label_encoder is not None,
        "sample_data": sample_data is not None,
    }

    overall_status = (
        "healthy"
        if all(
            artifacts_status[key] for key in ["model", "preprocessor", "label_encoder"]
        )
        else "degraded"
    )
    api_status = "operational" if overall_status == "healthy" else "limited"

    return HealthResponse(
        status=overall_status,
        model_loaded=model is not None,
        version="2.0.0",
        api_status=api_status,
        artifacts_status=artifacts_status,
        timestamp=pd.Timestamp.now().isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    artifacts_status = {
        "model": model is not None,
        "preprocessor": preprocessor is not None,
        "label_encoder": label_encoder is not None,
        "feature_names": feature_names is not None,
        "sample_data": sample_data is not None,
    }

    critical_artifacts = ["model", "preprocessor", "label_encoder"]
    critical_status = all(artifacts_status[key] for key in critical_artifacts)

    overall_status = "healthy" if critical_status else "unhealthy"
    api_status = "fully_operational" if critical_status else "limited_functionality"

    return HealthResponse(
        status=overall_status,
        model_loaded=model is not None,
        version="2.0.0",
        api_status=api_status,
        artifacts_status=artifacts_status,
        timestamp=pd.Timestamp.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, http_request: Request):
    """Enhanced prediction endpoint with comprehensive error handling."""

    # Log request details
    client_ip = http_request.client.host
    logger.info(f"🎯 Prediction request from {client_ip}")

    try:
        # Validate that critical artifacts are loaded
        if not all([model, preprocessor, label_encoder]):
            missing = []
            if model is None:
                missing.append("model")
            if preprocessor is None:
                missing.append("preprocessor")
            if label_encoder is None:
                missing.append("label_encoder")

            logger.error(f"❌ Missing critical artifacts: {missing}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service unavailable: Missing artifacts {missing}. Please ensure model training is completed.",
            )

        # Convert request to dict and preprocess
        request_data = request.dict()
        logger.debug(f"📝 Input data: {request_data}")

        # Enhanced preprocessing
        processed_input = enhanced_preprocess_input(request_data)

        # Make prediction with error handling
        try:
            prediction = model.predict(processed_input)[0]
            prediction_proba = model.predict_proba(processed_input)[0]
            logger.debug(
                f"🎯 Raw prediction: {prediction}, probabilities: {prediction_proba}"
            )
        except Exception as e:
            logger.error(f"❌ Model prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model prediction failed: {str(e)}",
            )

        # Process prediction results
        try:
            class_names = label_encoder.classes_
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            confidence = float(np.max(prediction_proba))

            logger.info(
                f"✅ Prediction: {predicted_class} (confidence: {confidence:.3f})"
            )
        except Exception as e:
            logger.error(f"❌ Error processing prediction results: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing prediction: {str(e)}",
            )

        # Create enhanced probability dictionary
        all_probabilities = {}
        try:
            for class_name, prob in zip(class_names, prediction_proba):
                all_probabilities[str(class_name)] = float(prob)
        except Exception as e:
            logger.warning(f"⚠️ Error creating probability dict: {e}")
            all_probabilities = {"unknown": 1.0}

        # Enhanced risk level determination
        if confidence > 0.85:
            risk_level = "high"
        elif confidence > 0.65:
            risk_level = "medium"
        else:
            risk_level = "low"

        risk_level_somali = SOMALI_MAPPINGS["risk_levels"].get(risk_level, risk_level)

        # Enhanced symptom analysis
        symptom_breakdown = enhanced_count_symptoms(request_data)

        # Get enhanced tips
        matching_tips = get_enhanced_matching_tips(
            predicted_class, symptom_breakdown, confidence
        )

        # Create comprehensive response
        response = PredictionResponse(
            predicted_label=str(prediction),
            predicted_class=predicted_class,
            confidence=confidence,
            risk_level=risk_level,
            risk_level_somali=risk_level_somali,
            num_symptoms=symptom_breakdown["total"],
            matching_tips=matching_tips,
            all_probabilities=all_probabilities,
            model_info={
                "model_type": type(model).__name__,
                "version": "2.0.0",
                "features_used": len(processed_input[0])
                if len(processed_input.shape) > 1
                else len(processed_input),
            },
            prediction_metadata={
                "symptom_breakdown": symptom_breakdown,
                "timestamp": pd.Timestamp.now().isoformat(),
                "processing_successful": True,
                "confidence_level": "high"
                if confidence > 0.8
                else "medium"
                if confidence > 0.6
                else "low",
            },
        )

        # Log successful prediction
        logger.info(
            f"✅ Prediction successful: {predicted_class} "
            f"(confidence: {confidence:.3f}, symptoms: {symptom_breakdown['total']})"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected prediction error: {str(e)}",
        )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_enhanced_model_info():
    """Get comprehensive model information."""
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        artifacts_loaded = {
            "model": model is not None,
            "preprocessor": preprocessor is not None,
            "label_encoder": label_encoder is not None,
            "feature_names": feature_names is not None,
            "sample_data": sample_data is not None,
        }

        # Get model metadata
        model_meta = {
            "model_type": type(model).__name__,
            "has_predict_proba": hasattr(model, "predict_proba"),
            "has_feature_importances": hasattr(model, "feature_importances_"),
            **model_metadata,
        }

        info = ModelInfoResponse(
            model_type=type(model).__name__,
            features_count=len(feature_names)
            if feature_names
            else getattr(model, "n_features_in_", 0),
            classes=list(label_encoder.classes_) if label_encoder else [],
            feature_groups=FEATURE_GROUPS,
            model_metadata=model_meta,
            artifacts_loaded=artifacts_loaded,
        )

        logger.info("📊 Model info retrieved successfully")
        return info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model information: {str(e)}",
        )


@app.get("/reload-artifacts")
async def reload_artifacts():
    """Endpoint to reload model artifacts."""
    try:
        logger.info("🔄 Reloading model artifacts...")
        success = load_model_artifacts()

        if success:
            return {
                "status": "success",
                "message": "Model artifacts reloaded successfully",
            }
        else:
            return {
                "status": "partial",
                "message": "Some artifacts could not be loaded",
            }

    except Exception as e:
        logger.error(f"❌ Error reloading artifacts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload artifacts: {str(e)}",
        )


@app.get("/debug/paths")
async def debug_paths():
    """Debug endpoint to check file paths and availability."""
    try:
        debug_info = {
            "current_directory": str(Path.cwd()),
            "model_artifacts_dir": str(MODEL_ARTIFACTS_DIR),
            "model_artifacts_exists": MODEL_ARTIFACTS_DIR.exists(),
            "available_files": {},
            "python_path": sys.path[:3],  # First 3 entries
        }

        # Check for files in various locations
        search_dirs = [
            MODEL_ARTIFACTS_DIR,
            Path("models"),
            Path("../models"),
            Path("artifacts/final_model"),
            Path("."),
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                files = list(search_dir.glob("*.pkl")) + list(
                    search_dir.glob("*.joblib")
                )
                debug_info["available_files"][str(search_dir)] = [
                    str(f.name) for f in files
                ]
            else:
                debug_info["available_files"][str(search_dir)] = "Directory not found"

        return debug_info

    except Exception as e:
        return {"error": str(e)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler."""
    logger.error(f"❌ HTTP {exc.status_code}: {exc.detail} for {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": pd.Timestamp.now().isoformat(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler."""
    logger.error(f"❌ Unhandled exception for {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "error_type": type(exc).__name__,
            "timestamp": pd.Timestamp.now().isoformat(),
            "path": str(request.url.path),
        },
    )


# Health check and monitoring endpoints
@app.get("/metrics")
async def get_metrics():
    """Get API metrics and statistics."""
    try:
        metrics = {
            "api_version": "2.0.0",
            "status": "operational"
            if all([model, preprocessor, label_encoder])
            else "degraded",
            "artifacts_loaded": {
                "model": model is not None,
                "preprocessor": preprocessor is not None,
                "label_encoder": label_encoder is not None,
                "feature_names": feature_names is not None,
                "sample_data": sample_data is not None,
            },
            "feature_groups": {
                "ordinal_features": len(FEATURE_GROUPS["ordinal"]),
                "binary_features": len(FEATURE_GROUPS["binary"]),
                "nominal_features": len(FEATURE_GROUPS["nominal"]),
                "total_features": sum(len(group) for group in FEATURE_GROUPS.values()),
            },
            "model_info": {
                "model_type": type(model).__name__ if model else None,
                "classes": len(label_encoder.classes_) if label_encoder else 0,
                "can_predict": model is not None and preprocessor is not None,
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        return metrics

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": str(e), "timestamp": pd.Timestamp.now().isoformat()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical Triage FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    logger.info(f"🚀 Starting FastAPI server on {args.host}:{args.port}")

    uvicorn.run(
        "fastapi_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True,
    )
