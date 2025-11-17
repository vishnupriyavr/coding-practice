from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load breast cancer dataset
data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2, random_state=42)

# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)

# Evaluate the model
accuracy = knn.score(x_test, y_test)
print(f'K-NN Classifier Accuracy: {accuracy * 100:.2f}%')

# Predict classes for first 5 test samples
print(knn.predict(x_test[:5]))
