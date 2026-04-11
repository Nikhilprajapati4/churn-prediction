import streamlit as st
import pandas as pd
import joblib

from src.preprocess import clean_data, feature_engineering, feature_encode

# ✅ Load model & scaler (already trained)
model = joblib.load(r"D:\Churn prediction\src\models\model.pkl")
scaler = joblib.load(r"D:\Churn prediction\src\models\scaler.pkl")

st.title("Customer Churn Prediction")

# 👉 User Inputs
tenure = st.number_input("Tenure", 0, 100)
monthly = st.number_input("Monthly Charges", 0.0, 10000.0)
total = st.number_input("Total Charges", 0.0, 100000.0)

# 👉 Predict Button
if st.button("Predict"):

    # ✅ Step 1: Create input dataframe
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total]
    })

    # ✅ Step 2: Apply SAME preprocessing
    input_data = clean_data(input_data)
    input_data = feature_engineering(input_data)   # creates AvgMonthlyCharge
    input_data = feature_encode(input_data)

    # ✅ Step 3: Scaling (use saved scaler ONLY)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge']

    # ensure all required numeric columns exist
    for col in num_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # ✅ Step 4: Match training columns
    model_cols = model.feature_names_in_
    input_data = input_data.reindex(columns=model_cols, fill_value=0)

    # ✅ Step 5: Prediction
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # ✅ Step 6: Output
    st.subheader("Result")
    st.write("Prediction:", "Churn" if pred == 1 else "No Churn")
    st.write("Churn Probability:", round(prob, 2))