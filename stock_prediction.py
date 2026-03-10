import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Seaborn styling
sns.set_theme(style="darkgrid")

# Get stock ticker from user
ticker = input("Enter stock ticker (Example: AAPL, TSLA, MSFT): ")

print("Downloading stock data...")
data = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# Use closing price
data = data[['Close']]

# Create future prediction column
forecast_days = 30
data['Prediction'] = data[['Close']].shift(-forecast_days)

# Features and labels
X = np.array(data.drop(['Prediction'], axis=1))[:-forecast_days]
y = np.array(data['Prediction'])[:-forecast_days]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
predictions = model.predict(X_test)

# Model error
mse = mean_squared_error(y_test, predictions)
print("Model Mean Squared Error:", mse)

# Predict future prices
future_X = np.array(data.drop(['Prediction'], axis=1))[-forecast_days:]
future_predictions = model.predict(future_X)

print("\nPredicted Prices for Next 30 Days:")
print(future_predictions)

# Graph 1: Historical Prices
plt.figure(figsize=(12,6))
sns.lineplot(x=data.index, y=data['Close'].values.flatten(), label="Historical Price")
plt.title(f"{ticker} Stock Price History")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

#  Graph 2: Predictions
future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1)

plt.figure(figsize=(12,6))
sns.lineplot(x=data.index, y=data['Close'].values.flatten(), label="Historical Price")
sns.lineplot(x=future_dates[1:], y=future_predictions, label="Predicted Price")

plt.title(f"{ticker} Stock Price Prediction (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
