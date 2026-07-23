import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# Train Model Function
def train_model(X_train, y_train, model_type="rf"):
    
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

    elif model_type == "xgb":
        model = xgb.XGBClassifier(
            objective='binary:logistic',  # Use 'multi:softprob' for multi-class tasks
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42)
    else:
        raise ValueError("Invalid model type. Use 'rf' or 'xgb'")

    # Train model
    model.fit(X_train, y_train ) 


    # Save model
    joblib.dump(model, r"D:\Projects\ML projects\Churn prediction\src\models\model.pkl")
    print("Model trained and saved successfully ✅")

    return model