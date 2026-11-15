"""SVM Classification
---------------------------------------------
Goal
Train an SVM model on a labeled dataset.
Use different kernels (linear, RBF) and compare
performance.
Visualize the decision boundary.
Evaluate the model using accuracy, precision, recall,
and AUC.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score

X, Y = make_classification(
    n_samples = 500,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_classes = 2,
    n_clusters_per_class = 1,
    class_sep = 1.5,
    random_state = 42
)

df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
df["Target"] = Y

print(df.head())
print(df["Target"].value_counts())

#Split the dataset in the training and the testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, stratify=Y
    )

#Scale the feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train SVM models
models  = {
    #For the Linear one
    'Linear': SVC(kernel='linear',C = 1.0, probability=True, random_state=42),
    #For the RBF one 
    'RBF': SVC(kernel='rbf',C = 1.0, gamma = 'scale', probability=True, random_state=42)
}
results = []


for name, model in models.items():
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    Y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    ac = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division = 0)
    auc = roc_auc_score(Y_test, Y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': ac,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc
    })


#function for the boundary
def plot_decision_boundary(model, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(Y_min, Y_max, 400)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = model.predict(grid_scaled).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, levels=2)
    ax.scatter(
        X[Y == 0, 0], X[Y == 0, 1],
        c='blue', label='Class 0', edgecolors='k', s=30
    )
    ax.scatter(
        X[Y == 1, 0], X[Y == 1, 1],
        c='red', label='Class 1', edgecolors='k', s=30
    )
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize = (12, 5))
plot_decision_boundary(models['Linear'], 'Linear SVM', axes[0])
plot_decision_boundary(models['RBF'], 'RBF SVM', axes[1])
plt.tight_layout()
plt.savefig('images/svm_decision_boundary.png', dpi=200, bbox_inches='tight')
plt.show()
results_df = pd.DataFrame(results)
print(results_df)