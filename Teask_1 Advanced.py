"""Random Forest Classification
---------------------------------------------

Train a Random Forest model and tune
hyperparameters (e.g., number of trees, max depth).
Evaluate the model using cross-validation and
classification metrics (precision, recall, F1-score).
Perform feature importance analysis to identify the
most important features in the dataset.
Tools: Python, scikit-learn, pandas, matplotlib.
"""
import matplotlib.pyplot as plt
import pandas as pd
# for the dataset generation
from sklearn.datasets import make_classification
# for the model
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, StratifiedKFold


x, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_repeated = 0,
    n_classes=2,
    n_clusters_per_class=2,
    class_sep=1.0,
    random_state=42,
    flip_y=0.01
)
# To convert the dataset x in the label y

feature = [f"Feature_{i}" for i in range(1, x.shape[1] + 1)]
df = pd.DataFrame(x, columns=feature)
df["Target"] = y

Target_counts = df["Target"].value_counts().sort_index()
print(df.head())
print(Target_counts)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

print("\n__________________________________Hyperparameters")
print(grid_search.best_params_)

cv_results = cross_validate(
    best_model, x_train, y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"],
    n_jobs = 1
)

print("\n__________________________________Cross-validation results:")


metrics = {
    "Precision": cv_results["test_precision"].mean(),
    "Recall": cv_results["test_recall"].mean(),
    "F1": cv_results["test_f1"].mean()
}

for name, score in metrics.items():
    print(f"{name}: {score:.4f}")


y_pred = best_model.predict(x_test)

#for the Measures the proportion of all predictions that are correct.
acc = accuracy_score(y_test, y_pred)
#Measures how many positive predictions
pre = precision_score(y_test, y_pred, zero_division=0)
#how amny positive cases as been identifies
rec = recall_score(y_test, y_pred, zero_division=0)
# for making the y pred and y tese in single score
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n__________________________________Test set evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Class 0", "Class 1"]
)
print("\n__________________________________Classification Report:")
print(report)

# Feature importance analysis
feature_importance = pd.DataFrame(
    {
        "Feature": feature,
        "Importance": best_model.feature_importances_,
    }
).sort_values(by="Importance", ascending=False)

print("\nTop 10 important features:")
print(feature_importance.head(10).to_string(index=False))

# Plot top features
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance["Feature"][:10][::-1],
    feature_importance["Importance"][:10][::-1],
    color="steelblue",
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances from Random Forest")
plt.tight_layout()

plt.savefig('images/random_forest_feature_importance.png', dpi=200, bbox_inches='tight')
plt.show()

"""
 Dataset Preview:
 The dataset contains 1200 rows and 21 columns.
 20 columns represent numerical features (Feature_1 to Feature_20).
 The Target column represents the class label (0 or 1).
 Each row represents one example used to train and evaluate the model.
 The dataset is synthetically generated using make_classification().

 Target Distribution:
 Class 0 contains 599 examples.
 Class 1 contains 601 examples.
 The target classes are almost perfectly balanced.
 A balanced dataset helps prevent the model from favouring one class
 simply because it has significantly more examples than the other.
 """