import os
import streamlit as st
import pandas as pd
import joblib

from preprocess import clean_data, feature_engineering, feature_encode, feature_scaling

# ── Paths (relative — works on any machine & Streamlit Cloud) ──────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
COLS_PATH   = os.path.join(BASE_DIR, "models", "train_columns.pkl")


# ── Cache model so it loads only once, not on every Streamlit rerun ────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    train_columns = joblib.load(COLS_PATH)   # list of columns saved during training
    return model, scaler, train_columns


# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Prediction")
st.markdown("Fill in the customer details and click **Predict**.")

col1, col2, col3 = st.columns(3)
with col1:
    tenure  = st.number_input("Tenure (months)", min_value=0, value=12)
with col2:
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
with col3:
    total   = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

if st.button("Predict", type="primary"):
    try:
        model, scaler, train_columns = load_artifacts()

        # Build a full input row with sensible defaults
        # FIX: Do NOT include 'Churn' — drop it before preprocessing
        input_data = pd.DataFrame([{
            'gender':          'Male',
            'SeniorCitizen':   0,
            'Partner':         'No',
            'Dependents':      'No',
            'tenure':          tenure,
            'PhoneService':    'Yes',
            'MultipleLines':   'No',
            'InternetService': 'DSL',
            'OnlineSecurity':  'No',
            'OnlineBackup':    'No',
            'DeviceProtection':'No',
            'TechSupport':     'No',
            'StreamingTV':     'No',
            'StreamingMovies': 'No',
            'Contract':        'Month-to-month',
            'PaperlessBilling':'Yes',
            'PaymentMethod':   'Electronic check',
            'MonthlyCharges':  monthly,
            'TotalCharges':    total,
        }])

        # Preprocessing pipeline
        input_data = clean_data(input_data)
        input_data = feature_engineering(input_data)
        input_data = feature_encode(input_data)
        input_data = feature_scaling(input_data, train=False, scaler=scaler)

        # FIX: Align columns to exactly what the model was trained on
        # Add any missing columns as 0, drop any extra columns
        input_data = input_data.reindex(columns=train_columns, fill_value=0)

        # Predict
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ This customer is likely to **Churn**  (Probability: {prob:.0%})")
        else:
            st.success(f"✅ This customer is likely to **Stay**  (Probability: {1-prob:.0%})")

        st.progress(float(prob), text=f"Churn probability: {prob:.0%}")

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}\n\nMake sure you have run train.py first and the models/ folder exists.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")