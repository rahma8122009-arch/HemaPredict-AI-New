import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Importing the object-oriented architectural pipeline blocks
from data_pipeline import DataIngestionPipeline, ClinicalFeatureEngineer
from inference_core import EnsembleInferenceCore

# 1. Page Settings with Professional Medical Visual Aesthetics
st.set_page_config(
    page_title="HemaPredict AI Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Injecting Slate Blue and Clean Gray styling configurations
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    h1 { color: #2C3E50; font-family: 'Helvetica Neue', Arial; }
    .stButton>button { background-color: #4A69BD; color: white; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 HemaPredict AI: Clinical Decision Support Dashboard")
st.subheader("Early-Stage Malignant Classification via Routine Complete Blood Counts")

st.markdown("---")

# Initialize backend pipeline layers
@st.cache_resource
def initialize_core_pipeline():
    ingestion = DataIngestionPipeline()
    engineer = ClinicalFeatureEngineer()
    inference = EnsembleInferenceCore()
    return ingestion, engineer, inference

ingestion_layer, feature_layer, inference_layer = initialize_core_pipeline()

# 2. Dual Data Input Section: Sidebar Inputs or CSV Uploads
st.sidebar.header("📋 Patient Laboratory Biomarkers")
input_mode = st.sidebar.radio("Select Input Methodology:", ("Manual Entry", "Batch CSV Upload"))

patient_df = pd.DataFrame()

if input_mode == "Manual Entry":
    # Capturing raw laboratory metrics
    neutrophils = st.sidebar.number_input("Neutrophils (absolute count)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
    lymphocytes = st.sidebar.number_input("Lymphocytes (absolute count)", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
    platelets = st.sidebar.number_input("Platelets (×10^3/µL)", min_value=0.0, max_value=1000.0, value=250.0, step=10.0)
    hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=14.0, step=0.2)
    rbc = st.sidebar.number_input("Red Blood Cells (RBC ×10^6/µL)", min_value=0.0, max_value=10.0, value=4.8, step=0.1)
    
    # Structural dataframe construction
    raw_data = {
        'Neutrophils': [neutrophils], 'Lymphocytes': [lymphocytes],
        'Platelets': [platelets], 'Hemoglobin': [hemoglobin], 'RBC': [rbc]
    }
    patient_df = pd.DataFrame(raw_data)

else:
    uploaded_file = st.sidebar.file_uploader("Upload Patient Digital Assays (CSV)", type=["csv"])
    if uploaded_file is not None:
        patient_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Batch Laboratory Data Preview", patient_df.head())

# 3. Execution and Real-Time Evaluation Pipeline
if not patient_df.empty:
    if st.button("🚀 Run Biomarker Analytics Pipeline"):
        with st.spinner("Executing Multi-Layer Optimization Framework..."):
            
            # Step I: Ingestion, Imputation and Scaling
            processed_df = ingestion_layer.process_data(patient_df, is_training=False)
            
            # Step II: Mathematical Feature Engineering Calculation
            final_features_df = feature_layer.generate_features(processed_df)
            
            # Step III: Ensemble Inference
            # For demonstration, evaluating the first record index
            evaluation_record = final_features_df.iloc[[0]]
            diagnostic_results = inference_layer.predict_risk(evaluation_record)
            
            # 4. Interactive Graphical Visualization Layer
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Diagnostic Classification Output")
                predicted_class = diagnostic_results['predicted_class']
                confidence_score = diagnostic_results['confidence'] * 100
                
                # Visual Risk Meter Color Logic Configuration
                if diagnostic_results['class_index'] == 0:
                    st.success(f"**Classification Status:** {predicted_class}")
                    st.info(f"**Pipeline Inference Confidence:** {confidence_score:.2f}%")
                elif diagnostic_results['class_index'] == 1:
                    st.error(f"**🚨 High Risk Alert:** {predicted_class}")
                    st.warning(f"**Pipeline Inference Confidence:** {confidence_score:.2f}%")
                else:
                    st.warning(f"**⚠️ Borderline Monitor Alert:** {predicted_class}")
                    st.info(f"**Pipeline Inference Confidence:** {confidence_score:.2f}%")
                    
                # Display individual distribution splits
                st.write("#### Probability Distribution Splits:")
                for k, v in diagnostic_results['probabilities'].items():
                    st.progress(float(v), text=f"{k}: {v*100:.1f}%")
            
            with col2:
                st.write("### 🧠 Game-Theoretic SHAP Interoperability Layer")
                st.write("Physiological feature attributions contributing to this classification outcome:")
                
                # Compute and display localized SHAP force properties
                shap_values = inference_layer.get_shap_values(evaluation_record)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                # Plot summary/bar for the specific sample classes
                shap.plots.bar(shap_values[0][:, 0], max_display=5, show=False)
                st.pyplot(plt.gcf())
                plt.clf()
                
            st.markdown("---")
            st.write("### 📋 Formulated Clinical Metrics Summary (Engineered Space)")
            st.dataframe(final_features_df)
            
            # Placeholder for Report Utility Export
            st.download_button(
                label="📥 Export Automated Clinical Report (PDF Placeholder)",
                data="HemaPredict AI Diagnostic Report Summary Data",
                file_name="HemaPredict_Diagnostic_Report.pdf",
                mime="text/plain"
            )
else:
    st.info("💡 Data Pipeline Input Required. Please adjust parameters in the left sidebar configuration panel and click Analyze.")