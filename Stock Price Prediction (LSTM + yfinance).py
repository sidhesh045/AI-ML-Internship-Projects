import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download stock data
df = yf.download("AAPL", start="2015-01-01", end="2023-01-01")

# Use closing prices
data = df["Close"].values.reshape(-1,1)

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(data_scaled, time_step)

# Reshape for LSTM [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train/Test split
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")

# Train
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Predict
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot
plt.figure(figsize=(10,6))
plt.plot(y_test_scaled, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.legend()
plt.show()
