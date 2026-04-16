import pandas as pd
from sklearn.linear_model import LinearRegression

# Multivariate data
data = {
    'Area (m2)': [50, 70, 90, 110, 130],
    'Rooms': [2, 3, 3, 4, 4],
    'Distance (km)': [10, 5, 3, 2, 1],
    'Price (million Rp)': [300, 420, 500, 600, 750]
}
df = pd.DataFrame(data)

# Separate features (X) and target (Y)
X = df[['Area (m2)', 'Rooms', 'Distance (km)']]
Y = df['Price (million Rp)']

# Train model
model = LinearRegression()
model.fit(X, Y)

# Regression coefficients
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}")

# Prediction for: 100m2, 3 rooms, 4 km from city center
prediction = model.predict(pd.DataFrame([[100, 3, 4]], columns=['Area (m2)', 'Rooms', 'Distance (km)']))
print(f"Predicted price: Rp {prediction[0]:.0f} million")