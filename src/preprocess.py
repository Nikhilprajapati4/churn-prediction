#import libraries

import pandas as pd  
import numpy as np
from sklearn.preprocessing import StandardScaler

#data import

#data cleaning 
# Drop Unnecessary Column
def clean_data(df) :
    df = df.copy()
    #drop costomer id
    if 'customerID' in df.columns :
        df.drop('customerID', axis=1, inplace=True)
        print("customer id is removed")
    # change dtype of total charges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # fill null values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Replace "No internet/phone service"
    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)

    return df

# Feature Engineering
def feature_engineering(df) :
    service_cols = [
        'PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]
    #number of services each costomer used
    df['NumServices'] = df[service_cols].replace({'Yes':1, 'No':0}).sum(axis=1)

    # Avg monthly charge
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)

    return df


# feature encoding for modal fitting

#binary mapping
def feature_encode(df):
    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1, 'No':0, 'Female':1, 'Male':0}).astype(int)

    df = pd.get_dummies(df, drop_first=True)

    return df

#feature scaling
def feature_scaling(df , train = True , scaler = None) :
    num_cols = ['tenure','MonthlyCharges','TotalCharges','AvgMonthlyCharge']

    if train :
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df , scaler
    
    else :
        df[num_cols] = scaler.transform(df[num_cols])
        return df

print("done")