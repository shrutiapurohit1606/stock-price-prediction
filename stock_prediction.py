import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Download stock data
print("Downloading stock data...")
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# 2. Keep only closing price
data = data[['Close']]

# 3. Predict 30 days into the future
future_days = 30
data['Prediction'] = data['Close'].shift(-future_days)

# 4. Prepare feature and target data
X = np.array(data.drop(['Prediction'], axis=1))[:-future_days]
y = np.array(data['Prediction'])[:-future_days]

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Test prediction
pred_test = model.predict(X_test)

# 8. Calculate error
mse = mean_squared_error(y_test, pred_test)
print("Model Mean Squared Error:", mse)

# 9. Predict future prices
future = np.array(data.drop(['Prediction'], axis=1))[-future_days:]
future_predictions = model.predict(future)

print("\nPredicted Prices for Next 30 Days:")
print(future_predictions)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(data['Close'])
plt.title("Stock Price History")
plt.xlabel("Days")
plt.ylabel("Price")

plt.subplot(1,2,2)
plt.plot(future_predictions)
plt.title("Predicted Next 30 Days")
plt.xlabel("Days")
plt.ylabel("Predicted Price")

plt.tight_layout()
plt.show()