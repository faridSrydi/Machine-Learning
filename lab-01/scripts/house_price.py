from sklearn.linear_model import LinearRegression
import numpy as np

# Data contoh: luas rumah (m2) vs harga (juta Rp)
X = np.array([[50], [70], [90]]) # Fitur
y = np.array([300, 420, 500])    # Label

# Latih model regresi linear
model = LinearRegression()
model.fit(X, y)

# Prediksi harga rumah 80 m2
prediksi = model.predict(np.array([[80]]))
print(f"Prediksi harga: Rp {prediksi[0]:.0f} juta")