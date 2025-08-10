# Updated Streamlit app (main file)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests
from pathlib import Path
import sys
import logging
from typing import Dict

# Add current directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import FEATURE_GROUPS, BINARY_MAP, SOMALI_MAPPINGS, ARTIFACTS_DIR
from src.logging_config import setup_logging
from src.visualization import (
    create_visualization_tabs,
    create_data_insights,
)  # Import visualization module

# Setup logging
logger = setup_logging("INFO", "logs/streamlit.log")

# Page configuration
st.set_page_config(
    page_title="Nidaamka Qiimeynta Caafimaadka | Medical Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .somali-text {
        font-size: 1.2rem;
        color: #2c3e50;
        font-weight: 500;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .tips-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .form-container {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .stSelectbox > div > div {
        min-height: 2.5rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-weight: bold;
        background-color: #1f77b4;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifacts():
    """Load model artifacts for local predictions."""
    try:
        artifacts_path = Path(ARTIFACTS_DIR) / "final_model"
        artifacts = {}

        # Load model
        model_files = list(artifacts_path.glob("*_production.pkl"))
        if model_files:
            artifacts["model"] = joblib.load(model_files[0])

        # Load other artifacts
        for artifact_name, file_name in [
            ("preprocessor", "preprocessor.pkl"),
            ("label_encoder", "label_encoder.pkl"),
            ("feature_names", "feature_names.pkl"),
        ]:
            file_path = artifacts_path / file_name
            if file_path.exists():
                artifacts[artifact_name] = joblib.load(file_path)

        return artifacts
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return {}


def create_input_form():
    """Create the organized input form for symptoms."""
    st.markdown(
        '<h2 class="somali-text">📝 Macluumaadka Bukaan-ka | Patient Information</h2>',
        unsafe_allow_html=True,
    )

    inputs = {}

    with st.form("symptom_form"):
        with st.container():
            st.markdown('<div class="form-container">', unsafe_allow_html=True)

            # Collapsible sections
            with st.expander("🩺 Calaamadaha Guud | General Symptoms", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    symptoms_group1 = [
                        ("qandho", "Qandho | Fever"),
                        ("qufac", "Qufac | Cough"),
                        ("madax_xanuun", "Madax Xanuun | Headache"),
                        ("caloosh_xanuun", "Caloosh Xanuun | Stomach Pain"),
                        ("daal", "Daal | Fatigue"),
                    ]

                    for key, label in symptoms_group1:
                        inputs[key] = st.selectbox(
                            label,
                            options=["maya", "haa"],
                            format_func=lambda x: "Maya | No"
                            if x == "maya"
                            else "Haa | Yes",
                            key=f"gen_{key}",
                        )

                with col2:
                    symptoms_group2 = [
                        ("matag", "Matag | Nausea"),
                        ("dhaxan", "Dhaxan | Rash"),
                        ("qufac_dhiig", "Qufac Dhiig leh | Cough with Blood"),
                        ("neeftu_dhibto", "Neef Dhibaato | Breathing Difficulty"),
                    ]

                    for key, label in symptoms_group2:
                        inputs[key] = st.selectbox(
                            label,
                            options=["maya", "haa"],
                            format_func=lambda x: "Maya | No"
                            if x == "maya"
                            else "Haa | Yes",
                            key=f"gen_{key}",
                        )

            with st.expander("⚡ Calaamado Kale | Other Symptoms", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    other_symptoms1 = [
                        ("iftiinka_dhibayo", "Iftiinka Dhibayo | Light Sensitivity"),
                        ("qoortu_adag_tahay", "Qoor Adag | Stiff Neck"),
                        ("lalabo", "Lalabo | Dizziness"),
                        ("shuban", "Shuban | Vomiting"),
                    ]

                    for key, label in other_symptoms1:
                        inputs[key] = st.selectbox(
                            label,
                            options=["maya", "haa"],
                            format_func=lambda x: "Maya | No"
                            if x == "maya"
                            else "Haa | Yes",
                            key=f"other_{key}",
                        )

                with col2:
                    other_symptoms2 = [
                        ("miisankaga_isdhimay", "Miisaan Dhimis | Weight Loss"),
                        ("qandho_daal_leh", "Qandho Daal leh | Fever with Fatigue"),
                        ("matag_dhiig_leh", "Matag Dhiig leh | Nausea with Blood"),
                        (
                            "ceshan_karin_qoyaanka",
                            "Ma Ceshi Karin Qoyaan | Cannot Feed Family",
                        ),
                    ]

                    for key, label in other_symptoms2:
                        inputs[key] = st.selectbox(
                            label,
                            options=["maya", "haa"],
                            format_func=lambda x: "Maya | No"
                            if x == "maya"
                            else "Haa | Yes",
                            key=f"other_{key}",
                        )

            with st.expander("📊 Heerka Calaamadaha | Symptom Severity", expanded=True):
                col1, col2 = st.columns(2)

                severity_options = {
                    "mild": "Yar | Mild",
                    "moderate": "Dhexdhexaad | Moderate",
                    "high": "Aad u Daran | High",
                }

                with col1:
                    severity1 = [
                        ("heerka_qandhada", "Heerka Qandhada | Fever Level"),
                        ("muddada_qandhada", "Muddada Qandhada | Fever Duration"),
                        ("madax_xanuun_daran", "Madax Xanuun Daran | Severe Headache"),
                        (
                            "muddada_madax_xanuunka",
                            "Muddada Madax Xanuunka | Headache Duration",
                        ),
                        ("muddada_qufaca", "Muddada Qufaca | Cough Duration"),
                    ]

                    for key, label in severity1:
                        inputs[key] = st.selectbox(
                            label,
                            options=list(severity_options.keys()),
                            format_func=lambda x: severity_options[x],
                            key=f"sev_{key}",
                        )

                with col2:
                    severity2 = [
                        ("muddada_xanuunka", "Muddada Xanuunka | Pain Duration"),
                        ("muddada_daalka", "Muddada Daalka | Fatigue Duration"),
                        ("muddada_mataga", "Muddada Mataga | Nausea Duration"),
                        ("daal_badan", "Daal Badan | Severe Fatigue"),
                        ("matag_daran", "Matag Daran | Severe Nausea"),
                    ]

                    for key, label in severity2:
                        inputs[key] = st.selectbox(
                            label,
                            options=list(severity_options.keys()),
                            format_func=lambda x: severity_options[x],
                            key=f"sev_{key}",
                        )

            with st.expander("👤 Macluumaad Kale | Other Information", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    # Age group options
                    age_options = {
                        "young": "Dhalinyaro | Young",
                        "middle_age": "Dhex da' | Middle Age",
                        "old": "Waayeel | Elderly",
                    }

                    inputs["da'da"] = st.selectbox(
                        "Da'da | Age Group",
                        options=list(age_options.keys()),
                        format_func=lambda x: age_options[x],
                        key="da_da",
                    )

                    # Cough type options
                    cough_options = {
                        "normal": "Caadi | Normal",
                        "dry": "Qallalan | Dry",
                        "wet": "Qoyan | Wet",
                        "bloody": "Dhiig leh | Bloody",
                    }

                    inputs["nooca_qufaca"] = st.selectbox(
                        "Nooca Qufaca | Cough Type",
                        options=list(cough_options.keys()),
                        format_func=lambda x: cough_options[x],
                        key="nooca_qufaca",
                    )

                with col2:
                    # Pain location options
                    pain_options = {
                        "chest": "Laab | Chest",
                        "abdomen": "Caloosh | Abdomen",
                        "head": "Madax | Head",
                        "back": "Dhabar | Back",
                        "limbs": "Lugaha/Gacmaha | Limbs",
                    }

                    inputs["halka_xanuunku_kaa_hayo"] = st.selectbox(
                        "Halka Xanuunku kaa Hayo | Pain Location",
                        options=list(pain_options.keys()),
                        format_func=lambda x: pain_options[x],
                        key="halka_xanuunku_kaa_hayo",
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button(
                "🔍 Qiimee | Analyze", use_container_width=True, type="primary"
            )

    return inputs, submitted


def validate_inputs(inputs: Dict) -> bool:
    """Validate that all required inputs are provided."""
    required_fields = [
        "qandho",
        "qufac",
        "madax_xanuun",
        "caloosh_xanuun",
        "daal",
        "matag",
        "dhaxan",
        "qufac_dhiig",
        "neeftu_dhibto",
    ]

    missing = [field for field in required_fields if inputs.get(field) is None]

    if missing:
        st.error(f"Fadlan buuxi goobaha muhiimka ah: {', '.join(missing[:3])}...")
        return False
    return True


def make_prediction_api(inputs: Dict) -> Dict:
    """Make prediction using FastAPI endpoint."""
    try:
        api_url = "http://localhost:8000/predict"
        response = requests.post(api_url, json=inputs, timeout=30)

        if response.status_code == 200:
            return response.json()
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to prediction API. Using local prediction...")
        return make_prediction_local(inputs)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def make_prediction_local(inputs: Dict) -> Dict:
    """Make prediction using locally loaded model."""
    try:
        artifacts = load_model_artifacts()

        if not all(
            key in artifacts for key in ["model", "preprocessor", "label_encoder"]
        ):
            st.error("Required model artifacts not found for local prediction.")
            return None

        # Preprocess input
        input_df = pd.DataFrame([inputs])

        # Handle feature name mapping
        if "da'da" not in input_df.columns and "da_da" in input_df.columns:
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

        # Apply preprocessing
        processed_input = artifacts["preprocessor"].transform(input_df[all_features])

        # Make prediction
        prediction = artifacts["model"].predict(processed_input)[0]
        prediction_proba = artifacts["model"].predict_proba(processed_input)[0]

        # Get class names and probabilities
        class_names = artifacts["label_encoder"].classes_
        predicted_class = artifacts["label_encoder"].inverse_transform([prediction])[0]
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
        num_symptoms = sum(
            1
            for k, v in inputs.items()
            if k in FEATURE_GROUPS["binary"] and str(v).lower() in ["haa", "yes", "1"]
        )

        # Generate basic tips
        matching_tips = [
            "Fadlan la tashii dhakhtarka si loo helo baaris dheeri ah.",
            "Raadi caawimaad caafimaad hadii calaamadahu sii daraan.",
            "Raacaba halka xanuunku u horumaro.",
        ]

        return {
            "predicted_label": str(prediction),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_level_somali": risk_level_somali,
            "num_symptoms": num_symptoms,
            "matching_tips": matching_tips,
            "all_probabilities": all_probabilities,
        }

    except Exception as e:
        st.error(f"Local prediction error: {e}")
        return None


def display_prediction_results(prediction: Dict):
    """Display prediction results in a clean layout."""
    if not prediction:
        return

    st.markdown(
        '<h2 class="somali-text">📊 Natiijada Qiimeynta | Analysis Results</h2>',
        unsafe_allow_html=True,
    )

    # Main results cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Qayb-qaybsiga | Classification",
            prediction["predicted_class"].upper(),
            help="Natiijada qiimeynta xaaladda caafimaadka",
        )

    with col2:
        st.metric(
            "Kalsoonida | Confidence",
            f"{prediction['confidence']:.1%}",
            help="Ixtimaalka saxda ah ee qiimeynta",
        )

    with col3:
        st.metric(
            "Heerka Halista | Risk Level",
            prediction["risk_level_somali"].upper(),
            prediction["risk_level"].capitalize(),
            help="Heerka degdegga ah ee daryeelka loo baahan yahay",
        )

    with col4:
        st.metric(
            "Tirada Calaamadaha | Symptoms Count",
            prediction["num_symptoms"],
            help="Tirada guud ee calaamadaha la soo sheegay",
        )

    # Risk level visualization
    risk_level = prediction["risk_level"]
    risk_class = f"risk-{risk_level}"

    risk_messages = {
        "low": "✅ Hoos - Xaalad caadi ah | Low Risk - Normal condition",
        "medium": "⚠️ Dhexe - U baahan kormeer dhakhtar | Medium Risk - Needs medical attention",
        "high": "🚨 Sare - Deg deg loo baahan yahay caawimaad | High Risk - Urgent medical attention needed",
    }

    st.markdown(
        f'<div class="{risk_class}">{risk_messages.get(risk_level, "Unknown risk level")}</div>',
        unsafe_allow_html=True,
    )

    # Probability distribution
    st.markdown("### 📈 Ihtimaalka Qayb-qaybsiga | Class Probabilities")

    prob_df = pd.DataFrame(
        [
            {"Class": k, "Probability": v}
            for k, v in prediction["all_probabilities"].items()
        ]
    ).sort_values("Probability", ascending=False)

    fig_prob = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        title="Probability Distribution",
        color="Probability",
        color_continuous_scale="viridis",
        text="Probability",
    )
    fig_prob.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_prob.update_layout(
        height=400,
        yaxis_tickformat=".0%",
        xaxis_title="Noocyada Xaaladaha | Condition Types",
        yaxis_title="Ihtimaalka | Probability",
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # Tips section
    st.markdown('<div class="tips-container">', unsafe_allow_html=True)
    st.markdown(
        '<h3 class="somali-text">💡 Talooyin | Recommendations</h3>',
        unsafe_allow_html=True,
    )

    for i, tip in enumerate(prediction["matching_tips"], 1):
        st.markdown(f"**{i}.** {tip}")

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main application function."""
    # Header
    st.markdown(
        '<h1 class="main-header">🏥 Nidaamka Qiimeynta Caafimaadka<br/>Medical Triage System</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.1rem; color: #666;">
                Nidaam casri ah oo isticmaala AI si loo qiimeeyo xaaladda caafimaad ee bukaanka
                <br/>
                <em>Modern AI-powered system for medical triage assessment</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("🔧 Habaynta | Settings")

        prediction_mode = st.selectbox(
            "Habka Saadaalinta | Prediction Mode",
            ["API", "Local"],
            help="Choose between API prediction or local model",
        )

        show_visualizations = st.checkbox(
            "Muuji Sawirada | Show Visualizations",
            value=True,
            help="Display analysis charts and graphs",
        )

        show_insights = st.checkbox(
            "Muuji Fahamka Xogta | Show Data Insights",
            value=True,
            help="Show statistics from historical data",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Macluumaad | Information")
        st.info(
            "Fadlan buuxi dhammaan goobaha si aad u hesho qiimeen sax ah.\n\n"
            "Please fill all fields to get accurate assessment."
        )

    # Main content
    inputs, submitted = create_input_form()

    if submitted:
        if not validate_inputs(inputs):
            return

        with st.spinner("🔄 Waa la qiimeynayaa... | Analyzing..."):
            if prediction_mode == "API":
                prediction = make_prediction_api(inputs)
            else:
                prediction = make_prediction_local(inputs)

            if prediction:
                display_prediction_results(prediction)
                logger.info(
                    f"Prediction made: {prediction.get('predicted_class', 'unknown')}"
                )

    # Visualizations section
    if show_visualizations:
        st.markdown("---")
        create_visualization_tabs()

    # Data insights section
    if show_insights:
        st.markdown("---")
        create_data_insights()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>© 2024 Medical Triage System | Nidaamka Qiimeynta Caafimaadka</p>
            <p>⚠️ Nidaamkan ma beddelayo talo dhakhtar. Marwalba la tashii takhaasusyahanka caafimaadka.</p>
            <p><em>This system does not replace medical advice. Always consult healthcare professionals.</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Streamlit app error: {e}")
