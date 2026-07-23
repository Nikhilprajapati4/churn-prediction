import os

import joblib
import pandas as pd
import streamlit as st

from src.preprocess import (
clean_data,
feature_engineering,
feature_encode
)

# ============================================================

# PATH CONFIGURATION

# ============================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__))

MODEL_DIR = os.path.join(
BASE_DIR,
"src",
"models"
)

# ============================================================

# LOAD MODEL AND TRAINING COLUMNS

# ============================================================
model_path = os.path.join(
    MODEL_DIR,
    "model.pkl"
)

columns_path = os.path.join(
    MODEL_DIR,
    "train_columns.pkl"
)

try:

    model = joblib.load(model_path)

    train_columns = joblib.load(columns_path)

except Exception as e:

    st.error(
        f"Error loading model files: {e}"
    )

    st.stop()
# ============================================================

# STREAMLIT PAGE CONFIGURATION

# ============================================================

st.set_page_config(
page_title="Customer Churn Prediction",
page_icon="📊",
layout="wide"
)

# ============================================================

# APPLICATION TITLE

# ============================================================

st.title(
"📊 Customer Churn Prediction"
)

st.write(
"Predict whether a customer is likely to leave the company."
)

st.divider()

# ============================================================

# SIDEBAR - CUSTOMER INFORMATION

# ============================================================

st.sidebar.header(
"Customer Information"
)

gender = st.sidebar.selectbox(
"Gender",
["Male", "Female"]
)

senior_citizen = st.sidebar.selectbox(
"Senior Citizen",
[0, 1]
)

partner = st.sidebar.selectbox(
"Partner",
["Yes", "No"]
)

dependents = st.sidebar.selectbox(
"Dependents",
["Yes", "No"]
)

tenure = st.sidebar.slider(
"Tenure",
min_value=0,
max_value=72,
value=12
)

phone_service = st.sidebar.selectbox(
"Phone Service",
["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
"Multiple Lines",
[
"Yes",
"No",
"No phone service"
]
)

internet_service = st.sidebar.selectbox(
"Internet Service",
[
"DSL",
"Fiber optic",
"No"
]
)

online_security = st.sidebar.selectbox(
"Online Security",
[
"Yes",
"No",
"No internet service"
]
)

online_backup = st.sidebar.selectbox(
"Online Backup",
[
"Yes",
"No",
"No internet service"
]
)

device_protection = st.sidebar.selectbox(
"Device Protection",
[
"Yes",
"No",
"No internet service"
]
)

tech_support = st.sidebar.selectbox(
"Tech Support",
[
"Yes",
"No",
"No internet service"
]
)

streaming_tv = st.sidebar.selectbox(
"Streaming TV",
[
"Yes",
"No",
"No internet service"
]
)

streaming_movies = st.sidebar.selectbox(
"Streaming Movies",
[
"Yes",
"No",
"No internet service"
]
)

contract = st.sidebar.selectbox(
"Contract",
[
"Month-to-month",
"One year",
"Two year"
]
)

paperless_billing = st.sidebar.selectbox(
"Paperless Billing",
["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
"Payment Method",
[
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
]
)

monthly_charges = st.sidebar.number_input(
"Monthly Charges",
min_value=0.0,
value=70.0
)

total_charges = st.sidebar.number_input(
"Total Charges",
min_value=0.0,
value=1000.0
)

# ============================================================

# CREATE INPUT DATAFRAME

# ============================================================

input_data = pd.DataFrame({

"gender": [gender],

"SeniorCitizen": [senior_citizen],

"Partner": [partner],

"Dependents": [dependents],

"tenure": [tenure],

"PhoneService": [phone_service],

"MultipleLines": [multiple_lines],

"InternetService": [internet_service],

"OnlineSecurity": [online_security],

"OnlineBackup": [online_backup],

"DeviceProtection": [device_protection],

"TechSupport": [tech_support],

"StreamingTV": [streaming_tv],

"StreamingMovies": [streaming_movies],

"Contract": [contract],

"PaperlessBilling": [paperless_billing],

"PaymentMethod": [payment_method],

"MonthlyCharges": [monthly_charges],

"TotalCharges": [total_charges]

})

# ============================================================

# DISPLAY CUSTOMER DATA

# ============================================================

st.subheader(
"Customer Details"
)

st.dataframe(
input_data,
use_container_width=True
)

# ============================================================

# PREDICTION

# ============================================================

if st.button(
"🔮 Predict Churn",
use_container_width=True
):
    if st.button("🔮 Predict Churn",use_container_width=True):
    try:
        processed_data = clean_data(
            input_data.copy()
        )

        processed_data = feature_engineering(
            processed_data
        )

        processed_data = feature_encode(
            processed_data
        )

        processed_data = processed_data.reindex(
            columns=train_columns,
            fill_value=0
        )

        prediction = model.predict(
            processed_data
        )

        probability = model.predict_proba(
            processed_data
        )[0][1]

        if prediction[0] == 1:

            st.error(
                "⚠️ Customer is likely to churn"
            )

        else:

            st.success(
                "✅ Customer is unlikely to churn"
            )

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

        st.progress(
            float(probability)
        )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
