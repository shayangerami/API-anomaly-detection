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
from NeuralNet import *

# Dataset labeled with logistic regression model
df_dir = r"C:\Users\shaya\OneDrive\Desktop\Python\API Project\API\remaining_behavior_with_logreg_predictions.csv"

def main():
    df = csv_file_reader(df_dir)
    # Drop duplicate columns
    df = df.drop(columns=["pred"], axis=1)
    
    # Uncomment below for testing
    # print(df.columns)
    # print(len(df))
    # print(df["classification"].value_counts())
    
    ''' Preprocessing '''
    # Separate categorical and numerical columns
    categorical_cols, numerical_cols = separator(df)
    
    # Determine which features to apply log transformation on
    log_features = [
        'inter_api_access_duration(sec)',
        'sequence_length(count)',
        'vsession_duration(min)',
        'num_sessions',
        'num_users',
        'num_unique_apis'
    ]
    
    # Apply log transformation to numerical features
    df_logTransformed = LogTransform(df, log_features)
    
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = split_data(df_logTransformed, 0.4) # Adjust test size if needed

    # Scale and encode features
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, label_encoders, label_encoder_target = scaler_encoder(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols)
    
    # Convert pandas DataFrames to numpy arrays for k-fold cross validation
    X_train_array = X_train_scaled.to_numpy()
    X_test_array = X_test_scaled.to_numpy()
    
    ''' Create and train neural network '''
    
    history, model, cv_scores = NN(X_train_array, y_train_encoded, X_test_array, y_test_encoded, n_splits=5)
    #tensorboard_process = launch_tensorboard()
    
    ''' Analyze feature importance '''
    # Get feature names from the original dataframe
    feature_names = df.drop(['classification'], axis = 1).columns.tolist()
    
    # Analyze feature importance
    importance_scores = featureImportance(model, X_test_array, y_test_encoded, feature_names)
    
    # Prepare data for tabulate
    table_data = []
    for feature, importance in importance_scores.items():
        table_data.append([feature, f"{importance:.4f}"])
    
    # Print feature importance in a table format
    print("\nFeature Importance Analysis:")
    print(tabulate(table_data, 
                  headers=['Feature', 'Importance Score'],
                  tablefmt='grid',
                  floatfmt=".4f"))
    
    # Keep TensorBoard running until user input
    input("Press Enter to stop TensorBoard and exit...")
    #tensorboard_process.terminate()
    
if __name__ == "__main__":
    main()