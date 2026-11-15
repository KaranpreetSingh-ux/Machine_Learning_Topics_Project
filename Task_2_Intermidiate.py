"""
Task 3: K-Means Clustering
Description: Implement K-Means clustering to group
unlabeled data into clusters (e.g., customer
segmentation).

Load a dataset and preprocess it (scaling).
Apply K-Means clustering and determine the optimal
number of clusters using the elbow method.
Visualize clusters using 2D scatter plots.
Interpret the clustering results.
Tools: Python, scikit-learn, matplotlib, seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


#for importing the file of the iris dataset
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, 'iris.csv'))

 #Data preprocessing 
#df = pd.read_csv(df = pd.read_csv('/Users/karanpreetsingh/Downloads/1) iris.csv'))

print("Dataset")
print(df.head())

X = df.drop(columns = ['species'])

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Elbow method 

inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init = 10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', color='blue')
plt.title("Elbow Method for Iris Dataset")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('images/kmeans_elbow.png', dpi=200, bbox_inches='tight')
plt.show() 

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init = 10)
df["cluster"] = kmeans.fit_predict(X_scaled)

#Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data = df,
    x=df["petal_length"],
    y=df["petal_width"],
    hue=df["cluster"],
    palette="Set2",
    s=100,
    edgecolor="pink"
)

plt.title("K-Means Clustering on Iris Dataset os here ")
plt.xlabel("Petal_Length")
plt.ylabel("Petal_Width")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig('images/kmeans_clusters.png', dpi=200, bbox_inches='tight')
plt.show()

print("Cluster Centers:")
print(df.groupby("cluster")[["sepal_length","sepal_width", "petal_length", "petal_width"]].mean())

print("\nCluster Counts vs actual spieces:")
print(pd.crosstab(df["cluster"], df["species"]))


"""
 Display the first five rows of the dataset to understand its structure.

 The elbow graph helps determine the optimal number of clusters by
 identifying the point where the decrease in WCSS starts to slow down.

 The scatter plot visualizes the clusters using petal length and petal width.
 Different colours represent different clusters identified by K-Means.

 Display the average flower measurements for each cluster to understand the characteristics of the flowers grouped together.

Compare the predicted clusters with the actual Iris species to understand
 how well the clusters correspond to the known species.
"""