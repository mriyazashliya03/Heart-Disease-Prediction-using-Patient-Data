# Heart-Disease-Prediction-using-Patient-Data

##  Team Members
- ASHLIYA M RIYAZ
- ARYA MOHAN G
- AISWARYA S


This project uses patient medical data to build and evaluate machine learning models for predicting heart disease, aiming to support early diagnosis and improve healthcare decision-making.

##  Dataset Description

The project uses the Cleveland Heart Disease dataset, which contains clinical data of patients.

- Total records: 297  
- Features: 13 clinical attributes  
- Target variable:
  - 0 → No heart disease  
  - 1 → Presence of heart disease  

### Key Features:
- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Maximum Heart Rate (thalach)
- Exercise Induced Angina (exang)
- Oldpeak
- Slope, CA, Thal

---

## Data Science Lifecycle

### 1. Problem Definition
Predict the presence of heart disease using patient data.

### 2. Data Collection
Dataset obtained from UCI / Kaggle.

### 3. Data Preprocessing
- Data cleaning and preparation  
- Feature scaling applied  

### 4. Exploratory Data Analysis (EDA)
- Feature distribution analysis  
- Correlation heatmap  

### 5. Feature Selection
- Mutual Information  
- Chi-Square Test  

### 6. Model Building
- K-Nearest Neighbors (KNN)  
- Logistic Regression  
- Gradient Boosting  

### 7. Model Evaluation
- Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  

### 8. Model Interpretation
- Feature importance  
- Model comparison

- ## Results
The models were evaluated using accuracy, precision, recall, and F1-score. Performance comparison was carried out among KNN, Logistic Regression, and Gradient Boosting.Models were trained and evaluated, and the final model was saved and used for prediction in the deployed application.

The models were evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score. Based on the evaluation, Logistic Regression was selected as the final model and saved for prediction.The selected Logistic Regression model is used in the application to predict the presence of heart disease based on patient input features.
## Our Project Workflow

1. Data Collection  
2. Data Preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature Selection  
5. Model Training  
6. Model Evaluation  
7. Model Deployment  

This pipeline ensures a structured approach to solving the heart disease prediction problem.
