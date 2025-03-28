# Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
import pandas as pd

from Preprocess import *

def DecTree(X_train, y_train):
    # Apply undersampling
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
    
def LogReg(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = LogisticRegression(random_state=42, penalty='l2', solver='liblinear')
    clf.fit(X_resampled, y_resampled)
    
    return clf
    
def XGB(X_train, y_train):
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

def SVC(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
  
    clf = svm.SVC(kernel='rbf',
               gamma=0.1,
               C=10.0)
               
    clf.fit(X_resampled, y_resampled)
    
    return clf

def RidgeReg(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
  
    clf = Ridge(alpha=1)
               
    clf.fit(X_resampled, y_resampled)
    
    return clf

def LassoReg(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
  
    clf = Lasso(alpha=0.0001)
               
    clf.fit(X_resampled, y_resampled)
    
    return clf

    
    
def ElasticReg(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = ElasticNet(alpha=0.5, l1_ratio=0.3)

    clf.fit(X_resampled, y_resampled)
    
    return clf

    
    
def RandForest(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
  
    clf = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    criterion='gini',  # Or 'entropy' for information gain (used for classification)
    max_depth=None,  # Maximum depth of the tree; can set a value for controlling overfitting
    min_samples_split=2,  # The minimum number of samples required to split an internal node
    min_samples_leaf=1,  # The minimum number of samples required to be at a leaf node
    max_features=5,  # Number of features to consider for splitting a node
    bootstrap=True,  # Whether bootstrap samples are used when building trees
    random_state=42,  # Seed for reproducibility
    n_jobs=-1  # Number of jobs to run in parallel, use -1 to use all processors
    )
               
    clf.fit(X_resampled, y_resampled)
    
    return clf

def GBM(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = GradientBoostingClassifier(n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42)

    clf.fit(X_resampled, y_resampled)
    
    return clf
    

def LGBM(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = LGBMClassifier(n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbose=-1)

    clf.fit(X_resampled, y_resampled)
    
    return clf


def ADA(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = AdaBoostClassifier(n_estimators=100,
    learning_rate=0.1,
    random_state=42)

    clf.fit(X_resampled, y_resampled)
    
    return clf

def CAT(X_train, y_train):
    X_resampled, y_resampled = under_sampler(X_train, y_train)
    
    clf = CatBoostClassifier(iterations=100,
    learning_rate=0.1,
    depth=3,
    verbose=0,
    random_seed=42)

    clf.fit(X_resampled, y_resampled)
    
    return clf
