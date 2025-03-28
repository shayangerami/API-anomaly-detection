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
    print(df.columns)
    print(len(df))