 
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load the Model and the Scaler
# We use try/except to catch loading errors early
try:
    model = pickle.load(open('heart_disease_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.set_page_config(page_title="HeartCare AI", page_icon="❤️")
st.title("🩺 Heart Disease Clinical Assistant")

# 2. Input Form
with st.form("medical_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        sex_display = st.selectbox("Sex", ["Male", "Female"])
        sex = 1 if sex_display == "Male" else 0
        cp_display = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp_display)
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 500, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])
    with col2:
        ecg_display = st.selectbox("Resting ECG Results", 
                           options=[0, 1, 2], 
                           format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang_display = st.selectbox("Pain/Angina after Exercise?", ["No", "Yes"])
        exang = 1 if exang_display == "Yes" else 0        
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1, 
                help="Measure of heart stress during exercise. Higher values usually indicate higher risk.")
        slope_display = st.selectbox("Peak Exercise ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
        slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
        slope = slope_map[slope_display]
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3], 
                help="This shows blood flow to the heart. 0 is the most common for heart disease patients.")
        thal_display = st.selectbox("Thallium Stress Test Result", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
        thal = thal_map[thal_display]

    submitted = st.form_submit_button("Analyze Risk")

# 3. Prediction Logic
if submitted:
    # Build a list of the 13 features in the EXACT order the model expects
    raw_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    # Convert to a 2D array (1 row, 13 columns)
    features_array = np.array(raw_features).reshape(1, -1)
    
    try:
        # Scale the data
        features_scaled = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(features_scaled)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error("### ⚠️ Result: High Risk of Heart Disease")
            st.write("Clinical indicators suggest a high probability of heart disease.")
        else:
            st.success("### ✅ Result: Low Risk Detected")
            st.write("Clinical indicators do not show significant risk factors.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("This usually happens if the input data shape doesn't match the model requirements.")
