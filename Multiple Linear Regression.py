import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv')

# 3. Define features (X) and target (y)
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 8. Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)  # TV, Radio, Newspaper

# 9. Optional: visualize actual vs predicted
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal reference line
plt.show()