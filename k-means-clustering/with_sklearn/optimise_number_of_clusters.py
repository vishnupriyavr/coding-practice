import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create synthetic data - 1000 samples with 4 centers
data, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=42)

# Visualize the synthetic data
plt.scatter(data[:, 0], data[:, 1], s=30)
plt.title("Synthetic Data")
#plt.show()

silhouette_scores = []
k_range = range(2, 11)  # Testing k from 2 to 10
inertia = []

# Function to compute KMeans and silhouette scores for different k values
def optimise_number_of_clusters(data):
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)
        print(f'For n_clusters = {k}, silhouette score is {score:.4f}')

# Plot silhouette scores
optimise_number_of_clusters(data)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for different k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

plt.plot(k_range, inertia, marker="x")
plt.show()

plt.plot(k_range, silhouette_scores, marker="*")
plt.show()