# Script to Analyze Stock Data using yfinance
# This script installs required libraries, retrieves stock data using yfinance, and performs analysis.
# Each section is explained in detail for better understanding.

# Step 1: Install yfinance library
# The yfinance library is essential for fetching stock data directly from Yahoo Finance.
# This line installs yfinance, and the -q flag ensures a quiet installation without unnecessary output.
!pip install -q yfinance

# Import necessary libraries
# Importing libraries that are essential for data manipulation, visualization, and analysis.

import yfinance as yf  # Importing yfinance to fetch stock data
import pandas as pd    # Importing pandas for data manipulation and handling
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import numpy as np  # Importing numpy for numerical operations

# Step 2: Fetch stock data
# Using yfinance to download historical data for a specific stock.
# In this example, we are fetching data for Apple Inc. (AAPL) from January 1, 2020, to January 1, 2023.

# Define the stock symbol and date range for data retrieval
ticker_symbol = "AAPL"  # Stock ticker for Apple Inc.
start_date = "2020-01-01"  # Start date for data retrieval
end_date = "2023-01-01"    # End date for data retrieval

# Download stock data using yfinance
# The yf.download function retrieves historical data for the given stock symbol within the specified date range.
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display the first few rows of the dataset
# This helps to understand the structure of the dataset and check if data was downloaded correctly.
print("First 5 rows of the dataset:")
print(stock_data.head())

# Step 3: Data Cleaning and Preprocessing
# Perform basic data cleaning and preprocessing steps, including handling missing values and filtering columns.

# Check for any missing values in the dataset
# Missing values can distort analysis, so it's crucial to identify and handle them.
missing_values = stock_data.isnull().sum()  # Count missing values in each column
print("Missing values in each column:\n", missing_values)

# Drop rows with missing values
# This step removes any rows that contain missing data, ensuring a complete dataset.
stock_data.dropna(inplace=True)

# Step 4: Analyzing Closing Prices
# Extract and plot the closing prices of the stock to analyze its trend over the specified period.

# Select the 'Close' column, which represents the closing price of the stock each day.
closing_prices = stock_data['Close']

# Plotting the closing prices
# This plot provides a visual representation of the stock's performance over time.
plt.figure(figsize=(12, 6))  # Set the figure size for better readability
plt.plot(closing_prices, label="Closing Price")  # Plot the closing prices
plt.title(f"{ticker_symbol} Stock Closing Prices")  # Title of the plot
plt.xlabel("Date")  # Label for the x-axis
plt.ylabel("Price in USD")  # Label for the y-axis
plt.legend()  # Display legend
plt.show()  # Render the plot

# Step 5: Calculate Moving Averages
# Moving averages help smooth out price data to identify the trend direction. 
# We will calculate the 20-day and 50-day moving averages.

# Calculate the 20-day moving average of the closing prices
# A short-term moving average can help detect short-term trends.
stock_data['20_MA'] = stock_data['Close'].rolling(window=20).mean()

# Calculate the 50-day moving average of the closing prices
# A longer moving average provides a view of the medium to long-term trend.
stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()

# Plotting the moving averages along with the closing prices
plt.figure(figsize=(12, 6))
plt.plot(closing_prices, label="Closing Price")
plt.plot(stock_data['20_MA'], label="20-Day MA", linestyle='--')  # Short-term moving average
plt.plot(stock_data['50_MA'], label="50-Day MA", linestyle='--')  # Long-term moving average
plt.title(f"{ticker_symbol} Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price in USD")
plt.legend()
plt.show()

# Step 6: Calculate Daily Returns
# Daily returns indicate the percentage change in stock price from one day to the next, showing short-term volatility.

# Compute the daily returns using the pct_change() function
stock_data['Daily_Return'] = stock_data['Close'].pct_change()

# Plotting daily returns
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Daily_Return'], label="Daily Return")
plt.title(f"{ticker_symbol} Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.legend()
plt.show()

# Step 7: Analyze Volatility
# Volatility is a measure of the amount by which a stock price fluctuates in a specific period. 
# We calculate the standard deviation of daily returns to gauge volatility.

# Calculate the rolling 20-day volatility of the stock
# Volatility is calculated as the standard deviation of daily returns over a 20-day period.
stock_data['20D_Volatility'] = stock_data['Daily_Return'].rolling(window=20).std()

# Plotting the 20-day volatility
plt.figure(figsize=(12, 6))
plt.plot(stock_data['20D_Volatility'], label="20-Day Volatility")
plt.title(f"{ticker_symbol} 20-Day Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# Step 8: Save the Processed Data
# Finally, save the processed data with moving averages and daily returns to a CSV file for future analysis or review.

# Save the data to a CSV file
output_file = f"{ticker_symbol}_processed_data.csv"
stock_data.to_csv(output_file)
print(f"Processed data saved to {output_file}")
