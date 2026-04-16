from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Data contoh: penghasilan vs pengeluaran pelanggan (dalam juta Rp)
X = np.array([
    [3, 2], [4, 3], [10, 8],
    [15, 12], [20, 18], [30, 25]
])

# Gunakan K-Means dengan 2 cluster
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.predict(X)

# Visualisasi
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.xlabel("Penghasilan")
plt.ylabel("Pengeluaran")
plt.title("Clustering Pelanggan")
plt.show()