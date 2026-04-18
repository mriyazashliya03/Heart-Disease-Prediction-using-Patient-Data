import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load the Model and the Scaler
# These files must be in the same folder on GitHub for the site to work!
try:
    model = pickle.load(open('heart_disease_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure .pkl files are in the repository.")

# 2. Page Configuration
st.set_page_config(page_title="HeartCare AI", page_icon="❤️")

st.title("🩺 Heart Disease Clinical Assistant")
st.markdown("---")
st.write("Enter the patient's clinical data below to calculate heart disease risk.")

# 3. Create Input Form
with st.form("medical_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 500, 200)
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

    submitted = st.form_submit_button("Analyze Risk")

# 4. Prediction Logic
if submitted:
    # We must match the exact number of features the model was trained on (13)
    # Filling default values (0) for columns like 'restecg' and 'slope' not in the form
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, 0, thalach, exang, oldpeak, 1, ca, 0]])
    
    # Scale the data using the saved scaler
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] # Get the % chance

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"### ⚠️ High Risk Detected")
        st.write(f"The model predicts a **{probability:.1%}** probability of heart disease.")
    else:
        st.success(f"### ✅ Low Risk")
        st.write(f"The model predicts a **{probability:.1%}** probability of heart disease.")

st.info("Disclaimer: This tool is for educational purposes and should not be used as medical advice.")
