import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf  # For reading stock data from Yahoo
from datetime import datetime  # For time stamps

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Show every column (no "..." truncation) and don't wrap to a fixed width
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ---------------------------------------------------------------------------
# 1. Configuration — atm we only use NVDA, in future just change TICKER
# ---------------------------------------------------------------------------
TICKER = 'NVDA'
COMPANY_NAME = 'NVIDIA'

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, f"{TICKER}_history.csv")

# LSTM training settings
WINDOW = 60            # days of history per training sample
TRAIN_SPLIT = 0.95     # fraction of data used for training
EPOCHS = 3
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# 2. Load data: from CSV if it exists, otherwise download ALL historical
#    data (from the first trading day up to today) and save it as CSV.
# ---------------------------------------------------------------------------
def load_stock_data(ticker: str, csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        print(f"Loading {ticker} data from {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading full {ticker} history from Yahoo Finance ...")
        df = yf.download(
            ticker,
            period="max",             # all historical data
            auto_adjust=False,        # keep a separate "Adj Close" column
            multi_level_index=False,  # keep flat columns for a single ticker
        )
        df.to_csv(csv_path)
        print(f"Saved {len(df)} rows to {csv_path}")
    return df


df = load_stock_data(TICKER, CSV_PATH)

print(df.tail(10))
print(df.describe())  # Summary stats
print(df.info())      # General info


# ---------------------------------------------------------------------------
# 3. Plot all data for NVDA: price, volume, moving averages, daily returns
# ---------------------------------------------------------------------------

# Moving averages of the adjusted close
ma_days = [10, 20, 50]
for ma in ma_days:
    df[f"MA for {ma} days"] = df['Adj Close'].rolling(ma).mean()

# Daily return in percent
df['Daily Return'] = df['Adj Close'].pct_change()

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

# Full closing-price history
df['Adj Close'].plot(ax=axes[0, 0])
axes[0, 0].set_ylabel('Adj Close')
axes[0, 0].set_xlabel(None)
axes[0, 0].set_title(f"Closing Price of {COMPANY_NAME}")

# Trading volume
df['Volume'].plot(ax=axes[0, 1])
axes[0, 1].set_ylabel('Volume')
axes[0, 1].set_xlabel(None)
axes[0, 1].set_title(f"Sales Volume for {COMPANY_NAME}")

# Moving averages
df[['Adj Close'] + [f"MA for {ma} days" for ma in ma_days]].plot(ax=axes[1, 0])
axes[1, 0].set_xlabel(None)
axes[1, 0].set_title(f"Moving Averages of {COMPANY_NAME}")

# Daily return distribution
df['Daily Return'].hist(bins=50, ax=axes[1, 1])
axes[1, 1].set_xlabel('Daily Return')
axes[1, 1].set_ylabel('Counts')
axes[1, 1].set_title(f"Daily Return Distribution of {COMPANY_NAME}")

fig.tight_layout()


# ---------------------------------------------------------------------------
# 4. Prepare data and train the LSTM model
# ---------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Use only the 'Close' column as a NumPy array
data = df.filter(['Close'])
dataset = data.values

# Number of rows to train the model on
training_data_len = int(np.ceil(len(dataset) * TRAIN_SPLIT))
print(f"Training on {training_data_len} of {len(dataset)} rows")

# Scale the data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Build the training set: each sample is WINDOW consecutive scaled closing
# prices, the target is the price on the following day
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []
for i in range(WINDOW, len(train_data)):
    x_train.append(train_data[i - WINDOW:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
# LSTMs expect (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(f"x_train shape: {x_train.shape}")

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Build the test set from the remaining data (with WINDOW days of overlap so
# the first test sample has a full window of history)
test_data = scaled_data[training_data_len - WINDOW:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(WINDOW, len(test_data)):
    x_test.append(test_data[i - WINDOW:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict and transform back to dollars
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Root mean squared error on the validation set
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f"RMSE: {rmse:.4f}")


# ---------------------------------------------------------------------------
# 5. Plot the results
# ---------------------------------------------------------------------------
train = data[:training_data_len]
valid = data[training_data_len:].copy()  # copy avoids SettingWithCopyWarning
valid['Predictions'] = predictions

plt.figure(figsize=(16, 6))
plt.title(f'{COMPANY_NAME} — LSTM Close Price Prediction (RMSE: {rmse:.2f})')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

# Show the valid and predicted prices
print(valid)

plt.show()
