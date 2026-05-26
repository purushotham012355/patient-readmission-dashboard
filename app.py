import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Patient Readmission Risk Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Patient Readmission Risk Dashboard")
st.markdown("Enter patient details below to predict 30-day readmission risk.")

# Sidebar inputs
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age Group (0-9 scale)", 0, 9, 5)
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 3)
num_medications = st.sidebar.slider("Number of Medications", 1, 81, 15)
number_inpatient = st.sidebar.slider("Prior Inpatient Visits", 0, 21, 1)
number_emergency = st.sidebar.slider("Prior Emergency Visits", 0, 76, 0)
number_outpatient = st.sidebar.slider("Prior Outpatient Visits", 0, 42, 0)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 132, 40)
num_procedures = st.sidebar.slider("Number of Procedures", 0, 6, 1)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 5)

# Predict button
if st.sidebar.button("🔍 Predict Risk"):

    # Build input array with 42 features (fill rest with 0s)
    input_data = np.zeros(42)
    input_data[0] = 2   # race
    input_data[1] = 1   # gender
    input_data[2] = age
    input_data[3] = 1   # admission_type_id
    input_data[4] = 1   # discharge_disposition_id
    input_data[5] = 1   # admission_source_id
    input_data[6] = time_in_hospital
    input_data[7] = num_lab_procedures
    input_data[8] = num_procedures
    input_data[9] = num_medications
    input_data[10] = number_outpatient
    input_data[11] = number_emergency
    input_data[12] = number_inpatient
    input_data[13] = 1  # diag_1
    input_data[14] = 1  # diag_2
    input_data[15] = 1  # diag_3
    input_data[16] = number_diagnoses

    input_df = pd.DataFrame([input_data], columns=[f'f{i}' for i in range(42)])

    # Load training columns
    df = pd.read_csv('cleaned_patient_data.csv')
    X = df.drop(columns=['readmitted'])
    input_df.columns = X.columns

    # Predict
    risk_proba = model.predict_proba(input_df)[0][1]
    risk_percent = round(risk_proba * 100, 1)

    # Risk tier
    if risk_percent >= 50:
        tier = "🔴 HIGH RISK"
        color = "red"
    elif risk_percent >= 25:
        tier = "🟡 MEDIUM RISK"
        color = "orange"
    else:
        tier = "🟢 LOW RISK"
        color = "green"

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Risk Score: **{round(risk_percent, 1)}%**")
        st.markdown(f"### Status: **:{color}[{tier}]**")
        st.progress(float(risk_proba))

    with col2:
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        st.markdown("### Top Risk Factors")
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(shap_values, input_df, plot_type="bar", 
                         max_display=5, show=False)
        st.pyplot(fig)
        # AI Care Plan
    st.markdown("---")
    st.markdown("### 🤖 AI Generated Care Plan")

    with st.spinner("Generating care plan..."):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""You are a clinical decision support assistant.
        
A diabetic patient has been assessed with the following profile:
- Risk Score: {risk_percent}% ({tier})
- Days in Hospital: {time_in_hospital}
- Number of Medications: {num_medications}
- Prior Inpatient Visits: {number_inpatient}
- Prior Emergency Visits: {number_emergency}
- Number of Diagnoses: {number_diagnoses}

Generate a concise, structured care plan with:
1. Immediate actions
2. Follow-up recommendations  
3. Lifestyle advice
4. Warning signs to watch for

Keep it brief, professional and actionable."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        care_plan = response.choices[0].message.content
        st.markdown(care_plan)