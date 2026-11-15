import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


base = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base, "House_Prediction.csv")

columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

df = pd.read_csv(csv_path, header = None, names = columns, sep = r"\s+")
print("Preview")
print(df.head())

df = df.dropna()
X = df.drop(columns = ['MEDV'])
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Model coefficients
print("\nIntercept (b0):", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation")
print(f"Mean Squared Error(MSE): {mse}")
print(f"R-squared (R^2): {r2}")



"""# ------------------------------------------------------------
 Task 2: Simple Linear Regression - House Price Prediction

 Aim: Predict house price (MEDV, in $1000s) from 13 features
 (crime rate, rooms, pollution, tax, etc.) using scikit-learn.

 Steps:
 1. Load the CSV (no header, so column names are given manually).
 2. Preprocess: dropna() removes rows with missing values.
 3. Split data: X = 13 features, y = MEDV (target).
 4. Train/test split: 80% training, 20% testing (random_state=42
    keeps the split the same every run).
 5. Train LinearRegression: finds the equation
    MEDV = b0 + b1*CRIM + b2*ZN + ... + b13*LSTAT
 6. Interpret coefficients: each one is the change in price for a
    1-unit increase in that feature (others held constant).
    Positive (e.g. RM) raises price; negative (e.g. LSTAT, NOX,
    CRIM) lowers it.
 7. Evaluate on test data:
    - MSE: average squared error (lower is better).
    - R^2: share of price variation explained (closer to 1 is better).

 Result: MSE ~ 24 (typical error ~ $4,900), R^2 ~ 0.67, so the
 model captures the main trends but misses some non-linear effects.
 ------------------------------------------------------------"""