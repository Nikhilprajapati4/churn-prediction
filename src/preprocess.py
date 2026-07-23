#import libraries

import pandas as pd  
import numpy as np

#data cleaning 
# Drop Unnecessary Column
def clean_data(df) :
    df = df.copy()
    #drop costomer id
    if 'customerID' in df.columns :
        df.drop('customerID', axis=1, inplace=True)
        print("customer id is removed")

    if 'gender' in df.columns :
        df.drop('gender', axis=1, inplace=True)
        print("gender is removed")

    if 'PhoneService' in df.columns :
        df.drop('PhoneService', axis=1, inplace=True)
        print("PhoneService is removed")

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

    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)


    service_cols = ['MultipleLines','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]
    #number of services each costomer used
    df['NumServices'] = df[service_cols].replace({'Yes':1, 'No':0}).sum(axis=1)

    #replace

    
    # Avg monthly charge
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)

    return df


# feature encoding for modal fitting

#binary mapping
def feature_encode(df):
    binary_cols = ["StreamingMovies","StreamingTV","TechSupport","DeviceProtection","OnlineBackup","OnlineSecurity","MultipleLines",'Partner','Dependents','PaperlessBilling','Churn']
    
    cat_cols = ['Contract','PaymentMethod','InternetService']

    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1, 'No':0, 'Female':1, 'Male':0}).astype(int)

    df = df.astype(int)
    
    return df

#feature scaling
#not need

print("done")
