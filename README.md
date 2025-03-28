# API Security Anomaly Detection System

## Overview
This project implements a neural network-based anomaly detection system for API security. It analyzes various API access patterns and behaviors to identify potential security threats and suspicious activities.

## Features
- Neural network-based anomaly detection
- Feature importance analysis
- Cross-validation for robust model evaluation
- TensorBoard integration for training visualization
- Class imbalance handling
- Comprehensive security metrics tracking

## Project Structure

```
├── API/              # API-related data and configurations
├── catboost_info/    # CatBoost model information and artifacts
├── logs/             # Training logs and TensorBoard data
├── NeuralNet.py      # Neural network implementation
├── Preprocess.py     # Data preprocessing utilities
├── Model.py          # Traditional ML model implementation and training (for phase1)
├── Evaluate.py       # Model evaluation and metrics
├── FileReading.py    # File reading utilities
├── main.py           # Main execution script
├── main2.py          # Main execution script for phase 2
├── requirements.txt  # Project dependencies
├── .gitignore       # Git ignore rules
├── .gitattributes   # Git attributes for LFS
└── README.md        # Project documentation
```

## Dependencies
- TensorFlow
- NumPy
- scikit-learn
- TensorBoard

## Installation
1. Clone the repository
2. Install required packages:
```bash
pip install tensorflow numpy scikit-learn
```

## Usage
1. Prepare your API access data
2. Run the preprocessing pipeline
3. Train the model:
```python
from NeuralNet import NN, featureImportance

# Train the model
history, model, cv_scores = NN(X_train, y_train, X_test, y_test)

# Analyze feature importance
importance = featureImportance(model, X_test, y_test, feature_names)
```

## Model Architecture
- Input layer → Dense(8) → Dense(4) → Dense(4) → Output layer
- ReLU activation for hidden layers
- Sigmoid activation for output layer
- Batch normalization and dropout for regularization
- L1/L2 regularization for weight control

## Training Process
- 5-fold stratified cross-validation
- Early stopping with 5 epochs patience
- Dynamic class weight balancing
- TensorBoard logging for visualization
- Multiple evaluation metrics (Accuracy, AUC, Precision, Recall)

## Feature Metadata

### High Importance Features
1. **num_users** (0.41)
   - Number of unique users per source
   - Primary indicator of potential security threats
   - High values suggest unauthorized access attempts

2. **num_sessions** (0.19)
   - Number of sessions per user/IP
   - Indicates persistence of potential attacks
   - Helps distinguish automated vs. manual attempts

3. **num_unique_apis** (0.15)
   - Number of different APIs accessed in a session
   - Shows breadth of potential attack
   - Indicates reconnaissance or scanning behavior

### Medium Importance Features
1. **sequence_length(count)** (0.08)
   - Number of API calls in a session
   - Indicates complexity of API calls
   - Longer sequences might suggest automated attacks

2. **vsession_duration(min)** (0.07)
   - Duration of an API session
   - Helps identify session hijacking or token abuse
   - Supports other security indicators

3. **inter_api_access_duration(sec)** (0.06)
   - Time gap between API calls
   - Helps identify automated bot attacks
   - Shows request timing patterns

### Low Importance Features
1. **api_access_pattern** (0.02)
   - Pattern of API calls
   - Complex to capture effectively
   - May need improved representation

2. **api_access_time** (0.02)
   - Time of API calls
   - Limited standalone importance
   - Could be more valuable with time-based patterns

## Security Indicators
The system monitors various suspicious behaviors including:
- Automated bot attacks
- Brute-force attempts
- Session hijacking
- Token abuse
- Credential stuffing
- Vulnerability probing
- IP spoofing
- Unauthorized access attempts

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Your chosen license]

## Contact
[Your contact information] 