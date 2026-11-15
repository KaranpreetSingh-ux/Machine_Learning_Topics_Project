# Machine Learning Internship Projects

This repository contains multiple machine learning tasks completed during the Codveda Technology internship. Each script demonstrates a different concept, including data preprocessing, regression, classification, clustering, and evaluation using Python libraries such as `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.

## Repository Overview

The project includes the following files:

- `Task_1_Basic.py` — Data preprocessing and feature scaling
- `Task_1 Intermidiate.py` — Logistic regression for binary churn prediction
- `Teask_1 Advanced.py` — Random forest classification with feature importance
- `Task_2Simple.py` — Simple linear regression for house price prediction
- `Task_2_Intermidiate.py` — K-Means clustering on the Iris dataset
- `Task_2_Advanced.py` — Support Vector Machine (SVM) classification

## Project Tasks

### 1) Task 1: Data Preprocessing for Machine Learning
File: `Task_1_Basic.py`

This task focuses on preparing raw data for machine learning models.

Topics covered:
- Handling missing values
- Encoding categorical variables
- Standardizing numerical features
- Train-test splitting
- Data inspection and preparation

Dataset used:
- `iris.csv`

### 2) Task 1: Logistic Regression for Binary Classification
File: `Task_1 Intermidiate.py`

This script builds a logistic regression model to predict customer churn.

Topics covered:
- Loading and exploring the dataset
- Cleaning target values
- Handling missing values
- One-hot encoding categorical features
- Train/test split with stratification
- Feature scaling using `StandardScaler`
- Model training and prediction
- Accuracy, precision, recall, F1-score, ROC-AUC
- Confusion matrix and classification report
- Coefficient interpretation and odds ratio
- ROC curve visualization

Dataset used:
- `churn-bigml-80-2.csv`

#### Result
![ROC Curve for Logistic Regression](images/logistic_regression_roc.png)

### 3) Task 1: Random Forest Classification
File: `Teask_1 Advanced.py`

This project trains a Random Forest classifier and evaluates its performance.

Topics covered:
- Synthetic dataset generation using `make_classification`
- Train/test split
- Hyperparameter tuning with `GridSearchCV`
- Cross-validation
- Accuracy, precision, recall, and F1-score
- Classification report
- Feature importance analysis
- Visualization of most important features

#### Result
![Random Forest Feature Importance](images/random_forest_feature_importance.png)

### 4) Task 2: Simple Linear Regression
File: `Task_2Simple.py`

This model predicts house prices using multiple numeric features.

Topics covered:
- Loading a CSV file without headers
- Train-test splitting
- Linear regression model training
- Interpreting coefficients
- Evaluating using MSE and R-squared

Dataset used:
- `House_Prediction.csv`

### 5) Task 2: K-Means Clustering
File: `Task_2_Intermidiate.py`

This task applies K-Means clustering to the Iris dataset.

Topics covered:
- Data preprocessing and scaling
- Elbow method for choosing `k`
- K-Means clustering
- Cluster visualization
- Comparing cluster results with actual species labels

Dataset used:
- `iris.csv`

#### Elbow Plot
![K-Means Elbow Method](images/kmeans_elbow.png)

#### Cluster Visualization
![K-Means Clusters](images/kmeans_clusters.png)

### 6) Task 2: SVM Classification
File: `Task_2_Advanced.py`

This project compares Linear and RBF SVM classifiers on a synthetic dataset.

Topics covered:
- Feature scaling
- Support Vector Machine classification
- Different kernel functions
- Accuracy, precision, recall, and AUC
- Visualization of decision boundary

#### Implementation Result
![SVM Decision Boundary](images/svm_decision_boundary.png)

## Technology Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Environment Setup

It is recommended to use a Python environment with the required libraries installed.

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

From the repository root, run any script using:

```bash
python "Task_1 Intermidiate.py"
```

Or:

```bash
python Task_1_Basic.py
```

Make sure the dataset files are present in the same folder as the Python scripts.

## Dataset Files

- `churn-bigml-80-2.csv` — customer churn binary classification dataset
- `iris.csv` — classic classification and clustering dataset
- `House_Prediction.csv` — regression dataset for house price prediction

## Notes

- Some filenames contain spaces or inconsistent naming (`Task_1 Intermidiate.py`, `Teask_1 Advanced.py`), so you may want to rename them later for cleaner project organization.
- Each script is designed as a stand-alone learning task and can be executed independently.

## Summary

This repository demonstrates foundational machine learning workflows, including:
- preprocessing
- feature engineering
- train/test evaluation
- classification and regression
- unsupervised clustering
- model interpretation and visualization

## License

This project is intended for educational and internship learning purposes.
