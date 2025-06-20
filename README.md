# Loan Prediction Machine Learning Project

## Overview
This project implements multiple machine learning algorithms to predict loan approval outcomes. The model compares the performance of various classification algorithms to determine the most effective approach for loan prediction.

## Problem Statement
Predicting loan approval is a critical task for financial institutions. This project aims to build and compare different machine learning models to accurately predict whether a loan application will be approved or rejected based on various applicant features.

## Dataset
The project uses loan application data containing features relevant to loan approval decisions. The target variable is binary (approved/rejected).

## Models Implemented
The project compares the following machine learning algorithms:

1. **Logistic Regression** - Linear classifier using logistic function
2. **Decision Tree** - Tree-based classifier using feature splits
3. **Random Forest** - Ensemble method using multiple decision trees
4. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
5. **Support Vector Machine (SVM)** - Kernel-based classifier

## Performance Results
Based on the model evaluation, here are the performance metrics sorted by F1 Score:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 78.86% | 75.96% | 98.75% | 85.87% |
| Random Forest | 77.24% | 75.49% | 96.25% | 84.62% |
| SVM | 65.04% | 65.04% | 100.00% | 78.82% |
| Decision Tree | 69.11% | 74.42% | 80.00% | 77.11% |
| KNN | 57.72% | 63.21% | 83.75% | 72.04% |

## Key Findings
- **Logistic Regression** achieved the best overall performance with an F1 score of 85.87%
- **Random Forest** came second with competitive performance (F1: 84.62%)
- **SVM** showed perfect recall (100%) but lower precision
- High recall scores across most models suggest good detection of positive cases
- The models show varying trade-offs between precision and recall

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd loan-prediction-project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter notebook:
```bash
jupyter notebook loan_predict.ipynb
```

2. Run all cells to:
   - Load and preprocess the data
   - Train multiple models
   - Compare performance metrics
   - View results

## Project Structure
```
loan-prediction-project/
├── loan_predict.ipynb    # Main notebook with model implementation
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── data/               # Dataset files (if applicable)
```

## Model Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1 Score**: Harmonic mean of precision and recall

## Future Improvements
- Feature engineering and selection
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Cross-validation for more robust evaluation
- Handling class imbalance if present
- Feature importance analysis
- Model interpretability using SHAP or LIME

## Notes
- The Logistic Regression model showed a convergence warning, suggesting the need for:
  - Increased iterations (`max_iter`)
  - Feature scaling/normalization
  - Different solver options
- Consider data preprocessing steps like scaling for better SVM and KNN performance

## Contributing
Feel free to contribute by:
- Adding new models or algorithms
- Improving data preprocessing
- Enhancing visualization
- Adding feature engineering techniques

