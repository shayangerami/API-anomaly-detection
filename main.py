# Import libraries
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate # To better see results in terminal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Import from other files within directory
from FileReading import *
from Preprocess import *
from Model import *
from Evaluate import *
from NeuralNet import *

# Ignore
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', 1000)       # Set the maximum width of the display
pd.set_option('display.max_rows', None)    # Display all rows (if needed)

# Data directories
json_file_supervised = r"C:/Users/shaya/OneDrive/Desktop/Python/API Project/API/supervised_call_graphs.json"
json_file_all = r"C:/Users/shaya/OneDrive/Desktop/Python/API Project/API/remaining_call_graphs.json"
bahvior_supervised = r"C:/Users/shaya/OneDrive/Desktop/Python/API Project/API/supervised_dataset.csv"
behavior_test = r"C:/Users/shaya/OneDrive/Desktop/Python/API Project/API/remaining_behavior_ext.csv"

def main():
    ''' Data Reading '''
    # API calls in json format
    df_api = json_file_reader(json_file_supervised)
    
    # Training data including behavioral features and hand labels
    df_behavior = csv_file_reader(bahvior_supervised)
    # Drop unnecessary cols and nulls
    df_behavior = df_behavior.drop(columns = ['Unnamed: 0', '_id'], axis = 1).dropna()
    
    # Test data including behavioral features and an unknown model predicted labels
    df_behavior_test = csv_file_reader(behavior_test)
    # Drop unnecessary cols and nulls
    df_behavior_test = df_behavior_test.drop(columns = ['Unnamed: 0', '_id', 'behavior'], axis = 1).dropna()
    # Rename target var
    df_behavior_test.rename(columns={'behavior_type': 'classification'}, inplace=True)
    # Remove other types
    df_behavior_test = df_behavior_test[df_behavior_test['classification'].isin(['normal', 'outlier'])]
    df_behavior_test = df_behavior_test[~df_behavior_test['ip_type'].isin(['private_ip', 'google_bot'])]
    
    # Combine API and behavior data on ID for potential need
    #df_combined = combiner(df_api, df_behavior)
    
    ''' Preprocessing '''
    categorical_cols, numerical_cols = separator(df_behavior)  
    
    
    # Determine which features to apply log transformation on
    log_features = [
        'inter_api_access_duration(sec)',
        'sequence_length(count)',
        'vsession_duration(min)',
        'num_sessions',
        'num_users',
        'num_unique_apis'
    ]
    # ======== Main data for training =======
    df_behavior_logtransformed = LogTransform(df_behavior, log_features)
    
    X_train, X_test, y_train, y_test = split_data(df_behavior_logtransformed, 0.25)
    
    # For testing
    print("\nModel is being trained on the following data:")
    print("X_train: ", len(X_train))
    print("y_train: ", len(y_train))
    print("Class distribution: \n", y_train.value_counts())
    print("Data is being undersampled for training to balance classes, 442 for both classes.")
    
    
    # scaler, label_encoders, label_encoder_target are being extracted from original function to then be passed into external data encoder function
    # Note that we are only applying transform to any test data so fitting most be learned and extracted during training phase
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, label_encoders, label_encoder_target = scaler_encoder(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols)
    
    # ======== External data for testing =======
    df_behavior_test_logtransformed = LogTransform(df_behavior_test, log_features)
    
    # Test sets from test data 
    X_external = df_behavior_test_logtransformed.drop(['classification'], axis=1)
    y_external = df_behavior_test_logtransformed['classification']
    
    X_external_scaled, y_external_encoded = scaler_encoder_external_data(X_external, y_external, numerical_cols, categorical_cols, scaler, label_encoders, label_encoder_target)
    
    # For testing
    print("\nModel is being tested on the following data:")
    print("X_test: ", len(X_external))
    print("y_test: ", len(y_external))
    print("Class distribution: \n", y_external.value_counts())
    print("=============================================================")

    model_list = [DecTree, RandForest, LogReg, RidgeReg, LassoReg, ElasticReg, SVC, XGB, GBM, LGBM, ADA, CAT]
    for model in model_list:
        print(f"\nEvaluation for {model.__name__}: ")
        clf = model(X_train_scaled, y_train_encoded)
        Evaluate(clf, X_external_scaled, y_external_encoded)
        print("\n===========================================================")
   
    # Based on the results, Logistic Regression is picked to move forward with
    # Using to_data function to add predictions to test data as a new column
    clf = LogReg(X_train_scaled, y_train_encoded)
    y_pred = Evaluate(clf, X_external_scaled, y_external_encoded)
    df = to_data(df_behavior_test, y_pred)
    df.to_csv("remaining_behavior_with_logreg_predictions.csv", index=False)
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
'''
 Next steps:
 
Observation: 
Every model is behaving exactly as it should for an anomaly detection system
it's prioritizing sensitivity to threats over reducing false alarms.
This is the right approach for security applications
 
Pick the best model : logreg
Add predition labels to data (only for matches)

Phase 2:
Train NN on bigger data
Evaluate and fine tune

Start deployment, learn and have fun:)

'''