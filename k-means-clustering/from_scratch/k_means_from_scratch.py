import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k=3):
        """
        K-Means Clustering from scratch.
        Initialize the KMeansClustering with the number of clusters k.
        """
        self.k = k
        self.centroids = None
    
    @staticmethod
    def euclidean_distance(data_point, centroids):
        """
        Compute the Euclidean distance between a data point and all centroids.
        """
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iterations=300):
        """
        Fit the K-Means model to the data X.
        Parameters:
        X : numpy array of shape (n_samples, n_features)
            The input data to cluster.
        max_iterations : int
            The maximum number of iterations to run the algorithm.
        """

        # Initialize centroids randomly within the range of the dataset
        n_samples, n_features = X.shape
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, n_features))
        
        # Iterate to refine centroids
        for _ in range(max_iterations):
            y = []

            for data_point in X:
                # Compute distances from data point to centroids
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                # Find minimum distance for each data point and assign cluster
                cluster_number = np.argmin(distances)
                y.append(cluster_number)
            
            y = np.array(y)

            # Get indices of data points in each cluster
            cluster_indices = []
            cluster_indices = [np.where(y == i) for i in range(self.k)]

            # Update centroids
            cluster_centroids = []
            for i, indices in enumerate(cluster_indices):
                # Compute new centroid as mean of assigned points
                if len(indices[0]) > 0:
                    cluster_centroid = np.mean(X[indices], axis=0)
                else:
                    cluster_centroid = self.centroids[i]  # Keep the old centroid if no points assigned
                
                cluster_centroids.append(cluster_centroid)

            if np.max(self.centroids - cluster_centroids) < 1e-6:
                break  # Convergence achieved
            
            # Update centroids for next iteration
            self.centroids = np.array(cluster_centroids)
        
        return y
    
    def predict(self, X):
        """
        Predict the closest cluster each data point in X belongs to.
        Parameters:
        X : numpy array of shape (n_samples, n_features)
            New data to predict.
        Returns:
        y : numpy array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        y = []
        for data_point in X:
            distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
            cluster_number = np.argmin(distances)
            y.append(cluster_number)
        return np.array(y)
    
# Create synthetic data - 100 random points in 2D
# With sklearn
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
random_points, data_point_cluster_labels = make_blobs(n_samples=100, n_features=2, centers=4)
#random_points = np.random.randint(0, 100, (300, 2))

kmeans = KMeansClustering(k=4)
y = kmeans.fit(random_points)

# Evaluate clustering performance using Adjusted Rand Index
ari = adjusted_rand_score(data_point_cluster_labels, y)
print(f"Adjusted Rand Index: {ari}")

plt.scatter(random_points[:, 0], random_points[:, 1], c=y, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering from Scratch')
plt.show()
        
    

