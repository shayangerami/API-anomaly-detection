# Import libraries
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Separate categorical and numericals values
def separator(df):
    '''
    Output is a list of columns
    '''
    categorical_cols = df.select_dtypes(include=['object']).columns.to_list()

    numerical_cols = df.select_dtypes(exclude=['object']).columns.to_list()
    
    return categorical_cols, numerical_cols
    
def LogTransform(df, features):
    """
    Used for skewed data
    Features must be numeric
    
    """
    # Copy to avoid modifying original data
    df_transformed = df.copy()
    for feature in features:
        df_transformed[feature] = np.log1p(df_transformed[feature])
    return df_transformed
    
def split_data(df, test_size):
    
    X = df.drop(['classification'], axis = 1)
    y = df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state=42)
    
    return X_train, X_test, y_train, y_test 
    
def scaler_encoder(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols):
    """
    Normalizes  numerical columns and
    encodes categorical columns
    
    Makes Sure to fit only on training data
    
    normal = 0
    outlier = 1
    
    default = 1 
    datacenter = 0
    
    E = 0
    F = 1
    
    """
    # Copy to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Initialize the scaler and label encoders
    scaler = StandardScaler()
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
    label_encoder_target = LabelEncoder()  # For encoding target labels
    
    # Fit and transform the training data
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train_scaled[numerical_cols])
    
    # Removing target column from categorical_cols because X data doesnt include target 
    # Using a different var (with _ at the end) to keep categorical cols unchanged
    categorical_cols_ = [col for col in categorical_cols if col != 'classification']
    
    for col in categorical_cols_:
        X_train_scaled[col] = label_encoders[col].fit_transform(X_train_scaled[col].astype(str))
    
    # Transform the test data (don't fit again)
    X_test_scaled[numerical_cols] = scaler.transform(X_test_scaled[numerical_cols])
    
    for col in categorical_cols_:
        X_test_scaled[col] = label_encoders[col].transform(X_test_scaled[col].astype(str))
    
    # Label encode the target variables
    y_train_encoded = label_encoder_target.fit_transform(y_train)
    y_test_encoded = label_encoder_target.transform(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, label_encoders, label_encoder_target
    
def scaler_encoder_external_data(X_external, y_external, numerical_cols, categorical_cols, scaler, label_encoders, label_encoder_target):
    """
    Scale and encode external data (features and target) using the same transformation 
    as the training data (do not fit on external data).
    """
    # Copy external data to avoid modifying original
    X_external_scaled = X_external.copy()
    
    # Apply the same scaling transformation to the numerical columns in external data (no fitting)
    X_external_scaled[numerical_cols] = scaler.transform(X_external_scaled[numerical_cols])
    
    # Removing target column from categorical_cols because X data doesnt include target 
    # Using a different var (with _ at the end) to keep categorical cols unchanged
    categorical_cols_ = [col for col in categorical_cols if col != 'classification']
    
    # Apply the same label encoding to the categorical columns in external data (no fitting)
    for col in categorical_cols_:
        X_external_scaled[col] = label_encoders[col].transform(X_external_scaled[col].astype(str))
    
    # Encode the target variable for external data (no fitting)
    y_external_encoded = label_encoder_target.transform(y_external)
    
    return X_external_scaled, y_external_encoded
   
         
def under_sampler(X_train, y_train):
    """
    Apply random undersampling to balance the classes
    """
    # Create undersampler with explicit sampling strategy
    undersampler = RandomUnderSampler(
        sampling_strategy='majority',  # This will reduce majority class to match minority class
        random_state=42
    )
    
    # Apply undersampling
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled