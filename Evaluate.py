# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from Preprocess import under_sampler

def Evaluate(model, X_test, y_test):
    '''
    The function automatically detects whether it's dealing with a classification 
    or regression model based on the model type and provides appropriate metrics 
    and visualizations.
    
    '''
    
    # Predicting
    y_pred = model.predict(X_test)
    
    Reg_list = ["ElasticNet", "Ridge", "Lasso"]
    if model.__class__.__name__ in Reg_list:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nRegression Metrics:\n")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
    else:
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Try to get probability predictions for ROC-AUC
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = None
        
        print("\nClassification Metrics:\n")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:\n")
        print(cm)
        '''
        Output format:
        [[TN, FP],
         [FN, TP]]
         
        '''
    
    return y_pred
    
def to_data(df, y_pred):
    """
    Assigns predicted labels to the dataframe as 'pred' 
    
    """
    label_mapping = {0: "normal", 1: "outlier"}
    
    df["pred"] = pd.Series(y_pred, index=df.index).map(label_mapping)
    
    return df
        