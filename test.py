#API Project test 

# read file jason
import json
import pandas as pd
import numpy as np
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

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', 1000)       # Set the maximum width of the display
pd.set_option('display.max_rows', None)    # Display all rows (if needed)


json_file_supervised = r"C:/Users/shaya/OneDrive/Desktop/Python/API Project/API/supervised_call_graphs.json"
json_file_all = r"C:/Users/shaya/OneDrive/Desktop/Python/API Project/API/remaining_call_graphs.json"
bahvior_supervised = r"C:/Users/shaya/OneDrive/Desktop/Python/API Project/API/supervised_dataset.csv"
behavior_test = r"C:/Users/shaya/OneDrive/Desktop/Python/API Project/API/remaining_behavior_ext.csv"


def json_file_reader(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    #return data
    return pd.DataFrame(data)
    
def csv_file_reader(file_path):
    df_behavior = pd.read_csv(file_path)
    return df_behavior
    
def combiner(df1, df2):
    df_combined = pd.merge(df1, df2, on='_id')
    df_combined = df_combined.drop_duplicates(subset='_id')
    
    return df_combined
    
df_api = json_file_reader(json_file_supervised)

df_behavior = csv_file_reader(bahvior_supervised)
df_behavior = df_behavior.drop(columns = ['Unnamed: 0', '_id'], axis = 1).dropna()

df_behavior_test = csv_file_reader(behavior_test)
df_behavior_test = df_behavior_test.drop(columns = ['Unnamed: 0', '_id', 'behavior'], axis = 1).dropna()
df_behavior_test.rename(columns={'behavior_type': 'classification'}, inplace=True)
df_behavior_test = df_behavior_test[df_behavior_test['classification'].isin(['normal', 'outlier'])]
print(df_behavior_test['classification'].value_counts())
#df_combined = combiner(df_api, df_behavior)

# There are some duplicate ids in df_api but we will drop duplicates
#print(df_api[df_api.duplicated(subset=['_id'], keep=False)])
#print(df_behavior[df_behavior['_id'] == 'a60aa35a-c8c7-343f-9b55-cc03e826a765'])

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
    df_transformed = df.copy()
    for feature in features:
        df_transformed[feature] = np.log1p(df_transformed[feature])
    return df_transformed
    
def scaler(df, numerical_cols, categorical_cols):
    """
    normal = 0
    outlier = 1
    
    default = 1 
    datacenter = 0
    
    E = 0
    F = 1
    
    """
    df_scaled = df.copy()
    
    
    scaler = StandardScaler()
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
   
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    
    for col in categorical_cols:
        df_scaled[col] = label_encoders[col].fit_transform(df_scaled[col].astype(str))
      
    return df_scaled
    
def split_data(df):
    
    X = df.drop(['classification'], axis = 1)
    y = df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y, random_state=42)
    
    return X_train, X_test, y_train, y_test
    
        
def under_sampler(X_train, y_train):
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

"""
def DecTree(X_train, y_train, X_test, y_test):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = DecisionTreeClassifier(class_weight = 'balanced', max_depth = 5, random_state=42, min_samples_split=20, min_samples_leaf=10)
    clf.fit(X_resampled, y_resampled)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    #cm = confusion_matrix(y_test, y_pred)
    
    importances = clf.feature_importances_
    for feature, importance in zip(X_train.columns, importances):
        print(f"{feature}: {importance:.4f}")
    
    return accuracy
"""
 
def DecTree(X_train, y_train, X_test, y_test):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = DecisionTreeClassifier(
    class_weight = 'balanced',
    max_depth = 5,
    random_state=42,
    min_samples_split=20,
    min_samples_leaf=10
    )
    clf.fit(X_resampled, y_resampled)
    
    return clf
"""
    
def LogReg(X_train, y_train, X_test, y_test):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = LogisticRegression(random_state=42, penalty='l2', solver='liblinear')
    clf.fit(X_resampled, y_resampled)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    #cm = confusion_matrix(y_test, y_pred)
    
    feature_importance = pd.DataFrame({
    'Feature': X_resampled.columns,
    'Coefficient': clf.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    
    print(feature_importance)
    print("accuracy; ", accuracy)
    
    return clf
"""
def LogReg(X_train, y_train, X_test, y_test):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = LogisticRegression(random_state=42, penalty='l2', solver='liblinear')
    clf.fit(X_resampled, y_resampled)
    
    return clf
    
def XGB(X_train, y_train, X_test, y_test):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
   
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        #use_label_encoder=False,
        random_state=42
    )
    clf.fit(X_resampled, y_resampled)
    
    return clf
    
def NN(X_train, y_train, X_test, y_test):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    y_resampled = y_resampled.values
    
    print("===================================")
    
    X_resampled = np.array(X_resampled, dtype=np.float32)  # Convert to float32 NumPy array
    y_resampled = np.array(y_resampled, dtype=np.float32) 
    
    
    model = Sequential([
    Dense(16, activation='relu', input_dim=X_resampled.shape[1]),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(8, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', 
                       tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
                       
    history = model.fit(X_resampled, y_resampled,
                    validation_split=0.2,
                    epochs=10,
                    batch_size=32,
                    verbose=1)   
    return history
    
def Evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    matches = (y_pred == y_test).sum()
    mismatches = (y_pred != y_test).sum()
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(accuracy)
    print(confusion_matrix(y_test, y_pred))
    
def main():
    categorical_cols, numerical_cols = separator(df_behavior)  
    
    log_features = [
        'inter_api_access_duration(sec)',
        'sequence_length(count)',
        'vsession_duration(min)',
        'num_sessions',
        'num_users',
        'num_unique_apis'
    ]
    df_behavior_logtransformed = LogTransform(df_behavior, log_features)
    df_behavior_test_logtransformed = LogTransform(df_behavior_test, log_features)

    df_behavior_logtransformed_scaled = scaler(df_behavior_logtransformed, numerical_cols, categorical_cols)
    df_behavior_test_logtransformed_scaled = scaler(df_behavior_test_logtransformed, numerical_cols, categorical_cols)
     
    X_train, X_test, y_train, y_test = split_data(df_behavior_logtransformed_scaled)

    # Numeric values average grouped by classification(0,1)
    #print(df_behavior.groupby('classification').mean(numeric_only=True))
    NN(X_train, y_train, X_test, y_test)

    X_external = df_behavior_test_logtransformed_scaled.drop(['classification'], axis=1)
    y_external = df_behavior_test_logtransformed_scaled['classification']


    #clf = XGB(X_train, y_train, X_test, y_test)
    
    #Evaluate(clf, X_external, y_external)
    
    
if __name__ == "__main__":
    main()

    
# Drop other classification types from df external