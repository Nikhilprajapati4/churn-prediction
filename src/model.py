import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Train Model Function
def train_model(X_train, y_train, model_type="rf"):
    
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

    elif model_type == "xgb":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    else:
        raise ValueError("Invalid model type. Use 'rf' or 'xgb'")

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, r"D:\Churn prediction\src\models/model.pkl")
    print("Model trained and saved successfully ✅")

    return model