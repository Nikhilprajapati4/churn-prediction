📊 Customer Churn Prediction

📌 Overview

This project predicts whether a customer will churn (leave the service) or not using Machine Learning.
It includes data preprocessing, feature engineering, model training, and deployment using Streamlit.

🚀 Features
Data Cleaning & Preprocessing
Feature Engineering
Feature Encoding & Scaling
Machine Learning Model (Classification)
Streamlit Web App for Prediction
🛠️ Tech Stack
Programming: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
Model Saving: Joblib
Web App: Streamlit
📂 Project Structure
Churn-Prediction/
│
├── data/
│   └── churn_data.csv
│
├── notebook/
│   ├── EDA.ipynb
│   ├── preprocess.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│       ├── model.pkl
│       └── scaler.pkl
│
├── app.py
├── requirements.txt
└── README.md
⚙️ Installation
# Clone repo
git clone https://github.com/your-username/churn-prediction.git

# Go to folder
cd churn-prediction

# Create environment (optional)
conda create -n churn_env python=3.10
conda activate churn_env

# Install dependencies
pip install -r requirements.txt
▶️ Run the Project
1. Train Model
python src/train.py
2. Run Streamlit App
streamlit run app.py
📊 Model Workflow
Data Cleaning
Feature Engineering (e.g., AvgMonthlyCharge, NumServices)
Encoding (categorical → numeric)
Feature Scaling
Model Training
Evaluation
📈 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
ROC-AUC
🧠 Use Case

Helps companies:
Reduce customer loss
Improve retention strategies
Identify high-risk customers.
company thinking about new impressive offers,
