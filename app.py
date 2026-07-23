import streamlit as st
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("src/models/model.pkl")
scaler = joblib.load("src/models/scaler.pkl")
model_cols = ['gender' 'SeniorCitizen' 'Partner' 'Dependents' 'tenure' 'PhoneService'
 'PaperlessBilling' 'MonthlyCharges' 'TotalCharges' 'NumServices'
 'AvgMonthlyCharge' 'MultipleLines_Yes' 'InternetService_Fiber optic'
 'InternetService_No' 'OnlineSecurity_Yes' 'OnlineBackup_Yes'
 'DeviceProtection_Yes' 'TechSupport_Yes' 'StreamingTV_Yes'
 'StreamingMovies_Yes' 'Contract_One year' 'Contract_Two year'
 'PaymentMethod_Credit card (automatic)' 'PaymentMethod_Electronic check'
 'PaymentMethod_Mailed check']

st.title("Customer Churn Prediction (Simple)")

# ===== Only 3 Inputs =====
tenure = st.number_input("Tenure (months)", min_value=0)
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")

if st.button("Predict"):

    # Default values for all required columns
    input_dict = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': tenure,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    df = pd.DataFrame([input_dict])

    # ===== Apply Pipeline =====
    df = clean_data(df)
    df = feature_engineering(df)
    df = feature_encode(df)

    # Align columns
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[model_cols]

    # Scaling
    df = feature_scaling(df, train=False, scaler=scaler)

    # Prediction
    pred = model.predict(df)[0]

    if pred == 1:
        st.error("Customer will Churn ❌")
    else:
        st.success("Customer will Stay ✅")
