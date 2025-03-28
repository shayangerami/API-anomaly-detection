# Import libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from Preprocess import *


def featureImportance(model, X_test, y_test, feature_names):
    """
    Analyze feature importance using multiple layers and activation patterns.
    
    Args:
        model: Trained neural network model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
    
    Returns:
        Dictionary of feature importance scores
    """
    # Get weights from all dense layers
    importance_scores = {}
    
    # Initialize scores for each feature
    for i in range(len(feature_names)):
        importance_scores[feature_names[i]] = 0
    
    # Consider weights from all dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights = layer.get_weights()[0]
            
            # For each feature
            for i in range(len(feature_names)):
                # Calculate importance based on absolute weights
                if i < weights.shape[0]:  # Only for input features
                    # Consider both positive and negative contributions
                    feature_importance = np.sum(np.abs(weights[i, :]))
                    importance_scores[feature_names[i]] += feature_importance
    
    # Normalize all scores to sum to 1
    total_importance = sum(importance_scores.values())
    if total_importance > 0:
        importance_scores = {k: v/total_importance for k, v in importance_scores.items()}
    
    
    # Sort features by importance
    sorted_importance = dict(sorted(importance_scores.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True))
    
    return sorted_importance


def create_model(input_dim, l1_l2_reg):
    """Create and compile the neural network model"""
    model = Sequential([
        Dense(16, activation='relu',  
              input_dim=input_dim,
              kernel_regularizer=l1_l2_reg),
        BatchNormalization(),
        Dropout(0.3), 
        
        
        Dense(8, activation='relu',  
              kernel_regularizer=l1_l2_reg),
        BatchNormalization(),
        Dropout(0.3), 
        
        Dense(1, activation='sigmoid',
              kernel_regularizer=l1_l2_reg)
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 
                          tf.keras.metrics.AUC(name='auc'),
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall')])
    return model


def NN(X_train, y_train, X_test, y_test, n_splits=5):
    """
    Train neural network with k-fold cross validation
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_splits: Number of folds for cross validation
    
    Returns:
        history: Training history
        model: Final trained model
        cv_scores: Cross validation scores
    """
    # Initialize k-fold cross validation with stratification
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize lists to store cross-validation scores
    cv_scores = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': []
    }
    
    # Ensure y_train and y_test are 2D arrays
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    
    # Define regularization parameters 
    l1_l2_reg = l1_l2(l1=0.01, l2=0.01)
    
    # Create a timestamp for unique log directory
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Power factor to control weight scaling (1.0 = linear, >1.0 = more aggressive)
    power_factor = 1
    
    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        print(f"\nFold {fold}/{n_splits}")
        print("-" * 50)
        
        # Split data for this fold
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Calculate class weights for this fold
        fold_total_samples = len(y_train_fold)
        fold_normal_samples = sum(y_train_fold == 0)
        fold_outlier_samples = sum(y_train_fold == 1)
        
        # Calculate weights with power factor for this fold
        fold_class_weights = {
            0: float((fold_total_samples / (2 * fold_normal_samples)) ** power_factor),  # minority class (normal)
            1: float((fold_total_samples / (2 * fold_outlier_samples)) ** power_factor)   # majority class (outlier)
        }
        
        # Add minimum weight threshold to prevent extreme values
        min_weight = 0.5
        max_weight = 2.0
        fold_class_weights = {
            k: float(max(min_weight, min(v, max_weight))) 
            for k, v in fold_class_weights.items()
        }
        
        # Normalize weights to sum to 2 (number of classes)
        total_weight = sum(fold_class_weights.values())
        fold_class_weights = {
            k: float(v * (2 / total_weight)) 
            for k, v in fold_class_weights.items()
        }
        
        print(f"\nClass Weights for Fold {fold}:")
        for k, v in fold_class_weights.items():
            print(f"Class {k}: {v:.4f}")
        
        # Create and compile model for this fold
        model = create_model(X_train.shape[1], l1_l2_reg)
        
        # Create fold-specific log directory
        fold_log_dir = f"logs/fit/{current_time}/fold_{fold}"
        
        # Add TensorBoard callback for this fold
        tensorboard_callback = TensorBoard(
            log_dir=fold_log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model with fold-specific class weights
        history = model.fit(X_train_fold, y_train_fold,
                          validation_data=(X_val_fold, y_val_fold),
                          class_weight=fold_class_weights,
                          epochs=50,
                          batch_size=32,
                          callbacks=[early_stopping, tensorboard_callback],
                          verbose=2)
        
        # Evaluate the model on validation set
        val_scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        
        # Store scores
        cv_scores['accuracy'].append(val_scores[1])
        cv_scores['auc'].append(val_scores[2])
        cv_scores['precision'].append(val_scores[3])
        cv_scores['recall'].append(val_scores[4])
    
    # Print cross validation results
    print("\nCross Validation Results:")
    for metric, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    # Train final model on all training data
    print("\nTraining final model on all training data...")
    final_model = create_model(X_train.shape[1], l1_l2_reg)
    
    # Calculate final class weights for the complete training set
    final_total_samples = len(y_train)
    final_normal_samples = sum(y_train == 0)
    final_outlier_samples = sum(y_train == 1)
    
    final_class_weights = {
        0: float((final_total_samples / (2 * final_normal_samples)) ** power_factor),
        1: float((final_total_samples / (2 * final_outlier_samples)) ** power_factor)
    }
    
    # Apply same normalization to final weights
    final_class_weights = {
        k: float(max(min_weight, min(v, max_weight))) 
        for k, v in final_class_weights.items()
    }
    
    total_weight = sum(final_class_weights.values())
    final_class_weights = {
        k: float(v * (2 / total_weight)) 
        for k, v in final_class_weights.items()
    }
    
    print("\nFinal Class Weights:")
    for k, v in final_class_weights.items():
        print(f"Class {k}: {v:.4f}")
    
    final_log_dir = f"logs/fit/{current_time}/final"
    final_tensorboard = TensorBoard(
        log_dir=final_log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    
    final_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    final_history = final_model.fit(X_train, y_train,
                                  validation_data=(X_test, y_test),
                                  class_weight=final_class_weights,
                                  epochs=50,
                                  batch_size=32,
                                  callbacks=[final_early_stopping, final_tensorboard],
                                  verbose=2)
    
    return final_history, final_model, cv_scores


def launch_tensorboard():
    """Launch TensorBoard in VS Code"""
    import subprocess
    import webbrowser
    import time
    
    # Start TensorBoard process
    process = subprocess.Popen(['tensorboard', '--logdir=logs/fit', '--port=6006'])
    
    # Wait for TensorBoard to start
    time.sleep(3)
    
    # Open TensorBoard in default browser
    webbrowser.open('http://localhost:6006')
    
    return process