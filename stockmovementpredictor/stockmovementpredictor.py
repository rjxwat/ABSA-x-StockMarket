# Script for Stock Price Analysis and Prediction Using GRU Model
# This script downloads stock data, processes it, calculates technical indicators, and trains a model to predict prices.

# Install required libraries (yfinance and ta for financial data and technical analysis indicators)
!pip install -q yfinance ta

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Download stock data from Yahoo Finance
# Specifying ticker symbols, including REliance for analysis and a date range
ticker_symbol = "RELIANCE.NS"  # Stock ticker for Reliance Industries
data = yf.download(ticker_symbol, start="2017-12-18", end="2023-11-09")

# Step 2: Data Cleaning and Preparation
# Load additional feature data and align dates
newstuff = pd.read_csv('/kaggle/input/aspect-feature/updated_aspect_feature.csv')
newstuff['date'] = pd.to_datetime(newstuff['date'])
newstuff.set_index('date', inplace=True)
newstuff.index = newstuff.index.tz_localize('UTC')
data = pd.concat([data, newstuff['Aspect_feature']], axis=1)
data.dropna(inplace=True)  # Remove rows with missing values

# Step 3: Calculate Technical Indicators
# Adding moving averages, price change, volatility, volume and trend indicators
data['ma_30'] = data['Close'].rolling('30D').mean()
data['price_change'] = data['Close'].diff()
data['price_change_pct'] = data['Close'].pct_change()
data['daily_range'] = data['High'] - data['Low']
data['daily_range_pct'] = data['daily_range'] / data['Close']
data['volume_ma5'] = data['Volume'].rolling(window=5).mean()
data['volume_ma20'] = data['Volume'].rolling(window=20).mean()
data['ma5'] = data['Close'].rolling(window=5).mean()
data['ma20'] = data['Close'].rolling(window=20).mean()
data['ma50'] = data['Close'].rolling(window=50).mean()

# Import ta library for additional indicators and calculate RSI and MACD
import ta
data['rsi'] = ta.momentum.rsi(data['Close'], window=14)
data['macd'] = ta.trend.macd_diff(data['Close'])

# Bollinger Bands calculation
data['bollinger_upper'] = data['ma_30'] + 2 * data['Close'].rolling(window=30).std()
data['bollinger_lower'] = data['ma_30'] - 2 * data['Close'].rolling(window=30).std()

# Step 4: Feature Selection and Correlation Analysis
# Identifying highly correlated features with 'Close' prices to improve model training
corr_matrix = data.corr()
top_features = corr_matrix['Close'].abs()[corr_matrix['Close'] > 0.6].index
X = data[top_features]
y = X

# Step 5: Data Scaling and Splitting
# Split data into training and testing sets (75% train, 25% test)
training_size = int(len(X) * 0.75)
X_train, X_test = X[:training_size], X[training_size:]
y_train, y_test = y[:training_size], y[training_size:]

# Apply MinMax scaling for normalization, which helps in faster convergence of the model
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Sequence Creation for Time Series Data
# Transform data into sequences for time series model input
def create_dataset(X, y, time_step=20):
    Xs, ys = [], []
    for i in range(len(X) - time_step - 1):
        Xs.append(X[i:(i + time_step)])
        ys.append(y[i + time_step])
    return np.array(Xs), np.array(ys)

# Convert scaled data into sequences
X_train_seq, y_train_seq = create_dataset(X_train_scaled, X_train_scaled)
X_test_seq, y_test_seq = create_dataset(X_test_scaled, X_test_scaled)

# Step 7: Model Creation and Training
# Define GRU model with four GRU layers and regularization to prevent overfitting
model = Sequential([
    GRU(256, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.3),
    GRU(128, return_sequences=True),
    Dropout(0.3),
    GRU(64, return_sequences=True),
    Dropout(0.3),
    GRU(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])

# Define an optimizer with an exponential decay schedule for learning rate
initial_learning_rate = 0.001
learning_rate_schedule = ExponentialDecay(initial_learning_rate, 1000, 0.9, staircase=True)
optimizer = Adam(learning_rate=learning_rate_schedule)

# Compile the model and set up early stopping to halt training if validation loss doesn’t improve
model.compile(optimizer=optimizer, loss='huber')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model using training data with 20% as validation and early stopping
history = model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Step 8: Model Predictions and Evaluation
# Predict on both train and test sequences to measure model performance
train_predict = model.predict(X_train_seq)
test_predict = model.predict(X_test_seq)

# Expand predictions for inverse scaling to original price range
def inverse_scale_and_extract(predicted, scaler, n_features):
    expanded = np.zeros((predicted.shape[0], n_features))
    expanded[:, 0] = predicted[:, 0]
    return scaler.inverse_transform(expanded)[:, 0]

# Inverse transform predictions and calculate error metrics (RMSE, R2, MAE)
train_actual = data['Close'][:len(train_predict)]
test_actual = data['Close'][len(train_predict):len(train_predict) + len(test_predict)]

train_rmse = mean_squared_error(train_actual, train_predict, squared=False)
test_rmse = mean_squared_error(test_actual, test_predict, squared=False)
train_r2 = r2_score(train_actual, train_predict)
test_r2 = r2_score(test_actual, test_predict)

# Display results
print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')
print(f'Train R-squared: {train_r2:.2f}')
print(f'Test R-squared: {test_r2:.2f}')
