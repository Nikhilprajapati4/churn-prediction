import pandas as pd
import joblib

from preprocess import clean_data, feature_engineering, feature_encode, feature_scaling

# load model
model = joblib.load(r"D:\Churn prediction\src\models\model.pkl")
scaler = joblib.load(r"D:\Churn prediction\src\models\scaler.pkl")

# 👉 load new data (or test data)
df = pd.read_csv(r"D:\Churn prediction\data\churn_data.csv")

# preprocessing (same steps)
df = clean_data(df)
df = feature_engineering(df)
df = feature_encode(df)
df , scaler = feature_scaling(df , train= False , scaler = scaler)

# features only
X = df.drop('Churn', axis=1)

# prediction
prediction = model.predict(X)
probability = model.predict_proba(X)[:,1]

# output
print(prediction[:5])
print(probability[:5])