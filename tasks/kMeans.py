import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = sns.load_dataset('iris')

data = data[data['species'] != 'versicolor']

X = data.iloc[:, :2]
y = data['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)


centroids = kmeans.cluster_centers_

fig, ax = plt.subplots()

scatter = ax.scatter(
    X_scaled[:, 0], X_scaled[:, 1], 
    c=clusters, cmap='Set1', 
    edgecolor='k', s=60, alpha=0.8
)

ax.scatter(
    centroids[:, 0], centroids[:, 1], 
    marker='X', c='black', s=250, 
    label='Centroids', edgecolor='white', linewidth=2
)

ax.set_title("K-Means Clustering", fontsize=14, fontweight='bold')
plt.legend(handles=scatter.legend_elements()[0], labels=['setosa', 'virginica'], title="Species")

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()