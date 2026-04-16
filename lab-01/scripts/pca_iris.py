from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset Iris (4 fitur -> direduksi jadi 2D)
data = load_iris()
X = data.data

# Reduksi dimensi dengan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualisasi
plt.scatter(X_pca[:,0], X_pca[:,1], c=data.target)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA pada Iris Dataset")
plt.show()