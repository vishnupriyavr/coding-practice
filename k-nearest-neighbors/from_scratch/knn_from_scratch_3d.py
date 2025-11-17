import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

points = {
    "blue": [[2,4,3], [1,3,5], [2,3,1], [3,2,3], [2,1,6]],
    "red":[[5,6,5], [4,5,2], [4,6,1], [6,6,1], [5,4,6], [10,10,4]]
}
query_point = [3,3,4]

class KNearestNeighbors:
    def __init__(self, k=4):
        """
        K-Nearest Neighbors from scratch.
        Initialize the KNearestNeighbors with the number of neighbors k.
        """
        self.k = k
        self.points = None

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        Compute the Euclidean distance between two points.
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def fit(self, points):
        self.points = points

    def predict(self, query_point):
        """
        Predict the class of the query_point based on k nearest neighbors.
        Parameters:
        query_point : numpy array of shape (n_features,)
            The input data point to classify.
        Returns:
        predicted_class : The predicted class label for the query_point.
        """
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = KNearestNeighbors.euclidean_distance(np.array(point), np.array(query_point))
                distances.append([distance, category])
        
        categories = [category for _, category in sorted(distances)[:self.k]]
        most_common = Counter(categories).most_common(1)
        return most_common[0][0]
    
# Example usage
knn = KNearestNeighbors(k=4)
knn.fit(points)
knn_predicted_class = knn.predict(query_point)

# Visualize points and query point
ax = plt.subplot(projection='3d')
ax.grid(True, c='lightgray', linestyle='--', linewidth=0.5)
ax.set_facecolor('black')
ax.tick_params(axis='x', color='white')
ax.tick_params(axis='y', color='white')

for point in points['blue']:
    ax.scatter(point[0], point[1], point[2], c='blue', s=60)

for point in points['red']:
    ax.scatter(point[0], point[1],point[2], c='red', s=60)

new_class_color = 'blue' if knn_predicted_class == 'blue' else 'red'
ax.scatter(query_point[0], query_point[1], query_point[2], c=new_class_color, s=100, marker='*', label='Query Point')

plt.title(f'K-NN Classification 3D (Predicted: {knn_predicted_class})')
plt.show()