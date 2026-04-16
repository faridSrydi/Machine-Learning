import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Actual and predicted data
Y = np.array([300, 420, 500, 600, 750])
Y_pred = np.array([320, 410, 490, 610, 740])

# Calculate MSE
mse = mean_squared_error(Y, Y_pred)
print(f"MSE: {mse:.2f}")

# Calculate RMSE
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")

# Calculate R2 Score
r2 = r2_score(Y, Y_pred)
print(f"R2 Score: {r2:.2f}")