# src/visualization.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.config import FEATURE_GROUPS, ARTIFACTS_DIR


def create_visualization_tabs():
    """Create visualization tabs for analysis results."""
    st.markdown(
        '<h2 class="somali-text">📊 Sawirada Qiimeynta | Analysis Visualizations</h2>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 Confusion Matrix",
            "📈 ROC Curves",
            "🔍 Precision-Recall",
            "⚡ SHAP Analysis",
        ]
    )

    artifacts_path = Path(ARTIFACTS_DIR) / "final_model" / "plots"

    with tab1:
        st.subheader("Confusion Matrix")
        confusion_files = list(artifacts_path.glob("*confusion_matrix.png"))
        if confusion_files:
            st.image(str(confusion_files[0]), use_column_width=True)
        else:
            st.info("Confusion matrix not available")

    with tab2:
        st.subheader("ROC Curves")
        roc_files = list(artifacts_path.glob("*roc_curve.png"))
        if roc_files:
            st.image(str(roc_files[0]), use_column_width=True)
        else:
            st.info("ROC curve not available")

    with tab3:
        st.subheader("Precision-Recall Curves")
        pr_files = list(artifacts_path.glob("*precision_recall.png"))
        if pr_files:
            st.image(str(pr_files[0]), use_column_width=True)
        else:
            st.info("Precision-recall curve not available")

    with tab4:
        st.subheader("SHAP Analysis")
        shap_files = list(artifacts_path.glob("*shap*.png"))
        if shap_files:
            for shap_file in shap_files:
                st.image(str(shap_file), use_column_width=True, caption=shap_file.stem)
        else:
            st.info("SHAP analysis not available")


def create_data_insights():
    """Create data insights section."""
    st.markdown(
        '<h2 class="somali-text">📈 Fahamka Xogta | Data Insights</h2>',
        unsafe_allow_html=True,
    )

    # Load sample data for insights
    try:
        data_path = Path("data/triage_data_cleaned.csv")
        if data_path.exists():
            sample_data = pd.read_csv(data_path)
        else:
            st.info("Sample data not available for insights")
            return
    except Exception:
        st.info("Sample data not available for insights")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        if "xaaladda_bukaanka" in sample_data.columns:
            class_counts = sample_data["xaaladda_bukaanka"].value_counts()
            fig_classes = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Distribution of Medical Conditions",
            )
            st.plotly_chart(fig_classes, use_container_width=True)

    with col2:
        st.subheader("Symptom Frequency")
        binary_cols = [
            col for col in FEATURE_GROUPS["binary"] if col in sample_data.columns
        ]
        if binary_cols:
            symptom_counts = []
            for col in binary_cols[:10]:
                count = (
                    (sample_data[col] == 1).sum()
                    if sample_data[col].dtype in [int, float]
                    else 0
                )
                symptom_counts.append({"Symptom": col, "Count": count})

            if symptom_counts:
                symptom_df = pd.DataFrame(symptom_counts)
                fig_symptoms = px.bar(
                    symptom_df,
                    x="Count",
                    y="Symptom",
                    orientation="h",
                    title="Most Common Symptoms",
                )
                st.plotly_chart(fig_symptoms, use_container_width=True)
