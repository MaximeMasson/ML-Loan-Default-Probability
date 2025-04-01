# Machine Learning Loan Default Probability Model

![GitHub last commit](https://img.shields.io/badge/last%20commit-April%202025-brightgreen)
![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project develops a sophisticated machine learning pipeline to predict the probability of loan defaults. Using a comprehensive dataset of loan characteristics and borrower information, the model identifies key risk factors and generates accurate default probability estimates that can be used for credit decisioning and risk management.

## Features

- **Modular Pipeline Architecture:** Structured as a series of independent, reusable components for data preprocessing, feature engineering, model training, and evaluation
- **Automated Feature Selection:** Implements correlation analysis and feature importance techniques to identify the most predictive variables
- **Model Comparison Framework:** Evaluates multiple machine learning algorithms against standardized performance metrics
- **Explainable AI Integration:** Provides transparent insights into model decisions through SHAP value analysis
- **Production-Ready Deployment:** Optimized model exported as joblib for seamless integration into production systems

## Project Structure

```
ML-Loan-Default-Probability/
├── data/                   # Data storage directory
├── model/                  # Saved model files
├── notebooks_class/        # Jupyter notebooks with clear workflow
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Model_Feedback_Loop.ipynb
│   ├── 04_Model_Insights.ipynb
│   ├── _Model_Comparator_class.py
│   └── _Model_Pipeline_class.py
└── conda-packages.txt      # Required packages
```

## Methodology

### 1. Data Preprocessing
The project begins with comprehensive data cleaning and preparation, including:
- Handling missing values through strategic imputation
- Feature normalization and scaling
- Encoding categorical variables
- Creating derived features from raw data

### 2. Exploratory Data Analysis
Visualizations and statistical analysis reveal key patterns in the data:
- Correlation analysis to identify relationships between features
- Distribution analysis to identify outliers and skewed variables
- Target variable exploration to understand class imbalance

### 3. Model Development
The core modeling approach employs:
- Gradient Boosting as the primary algorithm
- Automated hyperparameter optimization
- Cross-validation to ensure robustness
- Threshold optimization for precision/recall balance

### 4. Model Insights
The final model is analyzed to extract business value:
- Feature importance rankings to identify key risk drivers
- SHAP values to explain individual predictions
- Partial dependence plots to visualize feature relationships

## Results

The optimized Gradient Boosting model achieves:
- **AUC-ROC**: 0.92
- **Precision**: 0.87
- **Recall**: 0.79
- **F1 Score**: 0.83

Key predictive features include:
- Borrower's credit score (FICO)
- Debt-to-income ratio
- Loan amount relative to income
- Credit history length

## Applications

This model framework has direct applications in:
- Credit risk assessment for new loan applications
- Portfolio risk management and provisioning
- Stress testing for economic scenarios
- Regulatory compliance and capital adequacy

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
- joblib

## Future Development

- Integration of alternative data sources
- Real-time scoring API development
- Model monitoring and drift detection
- Fairness and bias analysis for responsible lending