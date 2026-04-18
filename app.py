%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load the Model and the Scaler
model = pickle.load(open('heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="HeartCare AI", page_icon="❤️")
st.title("🩺 Heart Disease Clinical Assistant")

# 2. Input Form
with st.form("medical_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        sex = st.selectbox("Sex (1=M, 0=F)", [1, 0])
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 500, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])
    with col2:
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina (1=Yes, 0=No)", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("ST Slope (1-3)", [1, 2, 3])
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (3=Normal, 6=Fixed, 7=Reversable)", [3, 6, 7])

    submitted = st.form_submit_button("Analyze Risk")

# 3. Prediction Logic
if submitted:
    # Create the 13-feature array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale and Predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.error("### ⚠️ Result: High Risk of Heart Disease")
    else:
        st.success("### ✅ Result: Low Risk detected")
