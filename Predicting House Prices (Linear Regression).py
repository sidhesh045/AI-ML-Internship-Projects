import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 3: Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 5: Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📌 House Price Prediction Results")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# -----------------------------
# Step 6: Visualization
# -----------------------------
plt.scatter(y_test, y_pred, color='green', alpha=0.5)
plt.xlabel("Actual House Prices ($100,000s)")
plt.ylabel("Predicted House Prices ($100,000s)")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference diagonal
plt.show()

