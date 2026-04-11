import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from preprocess import clean_data
from preprocess import feature_engineering
from preprocess import feature_encode
from preprocess import feature_scaling
from model import train_model
from evaluate import evaluate_model

# 1 Load data
df = pd.read_csv(r"D:\Churn prediction\data\churn_data.csv")

# 2 Preprocessing
df = clean_data(df)
df = feature_engineering(df)
df = feature_encode(df)
df, scaler = feature_scaling(df, train=True)

# 3 Split
X = df.drop('Churn', axis=1)
y = df['Churn']

# 4 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5 Train
model = train_model(X_train, y_train , model_type = "rf")

# Save scaler
joblib.dump(scaler, r"D:\Churn prediction\src\models\scaler.pkl")

# 6 Evaluate
evaluate_model(model, X_test, y_test)

print("Hurray! 🚀")