# Import necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Define the stock ticker and the date range
ticker = input("Enter Thicker Symbol:") # Apple Inc. You can replace this with any stock ticker
start_date = '2020-01-01'
end_date = '2023-01-01'

# Fetch the data
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Create the feature (previous day's close) and the label (next day's close)
stock_data['Previous_Close'] = stock_data['Close'].shift(1)
stock_data = stock_data.dropna()

# Define features and labels
X = stock_data[['Previous_Close']]
y = stock_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions= model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae= mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Plot the actual vs. predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Price')
plt.plot(y_test.index, predictions, label='Predicted Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'Stock Price Prediction for {ticker}')
plt.legend()
plt.show()
