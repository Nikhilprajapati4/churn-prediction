import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from preprocess import clean_data, feature_engineering, feature_encode
from model import train_model
from evaluate import evaluate_model

# ── Relative paths ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "churn_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)   # create models/ folder if missing

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(r'D:\Projects\ML projects\Churn prediction\data\churn_data.csv')

# ── 2. Preprocessing ───────────────────────────────────────────────────────────
df = clean_data(df)
df = feature_engineering(df)

# Drop Churn BEFORE encoding so it never gets encoded as a feature
y = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.drop('Churn', axis=1)
df = feature_encode(df)

# ── 3. Split ───────────────────────────────────────────────────────────────────
X = df
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train ───────────────────────────────────────────────────────────────────
model = train_model(X_train, y_train, model_type="xgb")

# ── 5. Save artifacts ──────────────────────────────────────────────────────────
joblib.dump(model,  os.path.join(MODEL_DIR, "model.pkl"))
# FIX: Save the exact column list so app.py can align prediction input
joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "train_columns.pkl"))
print("Model, scaler, and column list saved to models/")

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
evaluate_model(model, X_test, y_test)

print("Model and column list saved to models/")