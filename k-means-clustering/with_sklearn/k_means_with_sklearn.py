from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load and standardize the digits dataset
digits = load_digits()
data = StandardScaler().fit_transform(digits.data)

# K Means Clustering with sklearn
model = KMeans(n_clusters=10, random_state=42, init='random', n_init=10, max_iter=300)
model.fit(data)