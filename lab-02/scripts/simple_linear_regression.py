import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data: Land area (X) vs House price (Y)
X = np.array([50, 70, 90, 110, 130]).reshape(-1, 1)
Y = np.array([300, 420, 500, 600, 750])

# Train linear regression model
model = LinearRegression()
model.fit(X, Y)

# Predict price for 100 m2
prediction = model.predict(np.array([[100]]))
print(f"Predicted price for 100 m2: Rp {prediction[0]:.0f} million")

# Visualization
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Land Area (m2)')
plt.ylabel('House Price (million Rp)')
plt.legend()
plt.show()