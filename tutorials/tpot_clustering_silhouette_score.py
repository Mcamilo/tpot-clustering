import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# NOTE: Make sure that the csv file with the doesn't contain targets
training_features = pd.read_csv('PATH/TO/DATA/FILE', dtype=np.float64)

# Average CV score on the training set was: 0.8427159447547743

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(training_features)

# Perform PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
        
exported_pipeline = AgglomerativeClustering(linkage="ward", metric="euclidean", n_clusters=3)
clusters = exported_pipeline.fit_predict(pca_data)

# Plot PCA
plt.figure(figsize=(10, 7))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=100)
if hasattr(exported_pipeline,"cluster_centers_"):
    centroids = exported_pipeline.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, alpha=0.75)
plt.title('PCA of the Dataset with the exported clustering pipeline')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

# Save the PCA plot
plt.savefig('pca_plot.png')

print("PCA plot saved as 'pca_plot.png'.")
