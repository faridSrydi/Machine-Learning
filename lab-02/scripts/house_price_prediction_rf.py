import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Data Preparation
np.random.seed(42)
data = {
    'land_area': np.random.randint(50, 200, 100),
    'rooms': np.random.randint(1, 5, 100),
    'distance_to_center': np.round(np.random.uniform(1, 20, 100), 1),
    'year_built': np.random.randint(1990, 2023, 100)
}
df = pd.DataFrame(data)
df['price'] = (5 * df['land_area'] + 50 * df['rooms'] - 
               10 * df['distance_to_center'] - 
               0.5 * (2023 - df['year_built']) + 
               np.random.normal(0, 50, 100))

# 2. Split Data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), ['land_area', 'rooms', 'distance_to_center', 'year_built'])]
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 4. Training Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

# 5. Evaluation
y_pred = model.predict(X_test_processed)
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# 6. Feature Importance Visualization
importance = model.feature_importances_
features = X.columns
plt.barh(features, importance)
plt.title('Feature Importance for Price Prediction')
plt.show()