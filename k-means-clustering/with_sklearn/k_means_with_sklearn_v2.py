from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Load and standardize the digits dataset
digits = load_digits()
data = StandardScaler().fit_transform(digits.data)

# K Means Clustering with sklearn
model = KMeans(n_clusters=10, random_state=42, init='random', n_init=10, max_iter=300)
model.fit(data)

# Visualizing clusters
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
data_2d = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=model.labels_, cmap='tab10', s=15, alpha=0.7)
# Project cluster centers to PCA space and plot
centers_2d = pca.transform(model.cluster_centers_)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=100, marker='x', label='centroids')
plt.title('K-Means Clustering (PCA projection)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()