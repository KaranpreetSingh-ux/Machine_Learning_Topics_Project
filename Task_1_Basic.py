"""Task 1: Data Preprocessing for
Machine Learning
Description: Preprocess a raw dataset to make it ready
for machine learning.
Handle missing data (e.g., filling with mean/median,
dropping).
Encode categorical variables (e.g., using one-hot
encoding or label encoding).
Normalize or standardize numerical features.
Split the dataset into training and testing sets.
Tools: Python, pandas, scikit-learn.
"""

#Libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("iris.csv")

print(df.head())

numeric_cols = df.select_dtypes(include = ["number"]).columns
categorical_cols = df.select_dtypes(exclude = ["number"]).columns

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

for col in categorical_cols:
    if df[col].isnull().sum() >0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nAfter the missing values")

print(df.head())

X = df.drop(columns=["species"])
y = df["species"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#Split the data set
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

print("\nSample: ")
print(X_train_scaled[:5])