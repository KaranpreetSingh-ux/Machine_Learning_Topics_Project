"""
Task 1: Logistic Regression for
Binary 
Description: Implement a logistic regression model to
predict binary outcomes (e.g., whether a customer
will churn).
Classification

Load and preprocess the dataset.
Train a logistic regression model using scikit-learn.
Interpret model coefficients and the odds ratio.
Evaluate the model using metrics such as accuracy,
precision, recall, and the ROC curve.
Tools: Python, pandas, scikit-learn, matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, f1_score, classification_report

url = "churn-bigml-80-2.csv"
df = pd.read_csv(url)

df = df.rename(columns={'Churn': 'churn'})

print("Dataset:")
print(df.head())
print("\n Dataset shape:", df.shape)
print("\nColumn Information:")
df.info()
print("\n Missing Values:", df.isnull().sum())
print("\n churn Distribution:", df['churn'].value_counts())
print("\n Target Distribution:", df['churn'].value_counts(normalize=True))

df['churn'] = df['churn'].astype(str).str.strip().str.lower()
df['churn'] = df['churn'].map({'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0})

invalid_target = df['churn'].isna()
if invalid_target.any():
    print(f"Removing {invalid_target.sum()} invalid values.")
    df = df.loc[~invalid_target].copy()

#FOR THE TARGET VALUE
print("Converted Target distribution: ", df['churn'].value_counts())

y = df['churn'].astype(int)
X = df.drop(columns=['churn'])

numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X = X.fillna("Missing")

#For the categorial column to an dummy one 
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

#Prediction for the LR
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
print ("\nModel Evaluation Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall: {rec:.4f}")
print(f"AUC: {auc:.4f}")

print("\n Confusion matrix", confusion_matrix(y_test, y_pred))

print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print("Missing target values:", df['churn'].isna().sum())

if df['churn'].isna().any():
    raise ValueError("Unexpected values found in the churn column.")

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

coef = model.coef_[0]
odds_ratios = np.exp(coef)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coef,
    'Odds_Ratio' : odds_ratios
})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
print("\n Top Coefficient and Odd ratios:")
print(coef_df.head(10).drop(columns='Abs_Coefficient').to_string(index=False))
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})', color = "blue")
plt.plot([0, 1], [0, 1], linestyle='--', color = 'red', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for logistic regression is done')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/logistic_regression_roc.png', dpi=200, bbox_inches='tight')
plt.show()