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
#
# GPU: TensorFlow >= 2.11 is CPU-only on native Windows. To train on the
# NVIDIA GPU, run this script through WSL2:
#   wsl -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/Users/steve/PycharmProjects/rechnernetze/hist_test_all/run_wsl_gpu.sh > /tmp/run_gpu_hta.sh && bash /tmp/run_gpu_hta.sh"
# ---------------------------------------------------------------------------
TICKER = 'NVDA'
COMPANY_NAME = 'NVIDIA'

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, f"{TICKER}_history.csv")

# LSTM training settings
WINDOW = 60            # days of history per training sample
HORIZON = 21           # forecast horizon in trading days (~1 month)
TRAIN_SPLIT = 0.95     # fraction of data used for training
MAX_EPOCHS = 50        # early stopping usually ends training well before this
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
#
# The model predicts the CUMULATIVE LOG RETURN over the next HORIZON days
# ("direct" multi-step strategy: one shot at the whole horizon instead of
# chaining daily predictions, so forecast errors don't accumulate).
# Returns instead of price levels because prices are non-stationary (NVDA
# spent most of its history under $10, so a price-level model has almost no
# training signal near today's all-time highs). The predicted price is
# reconstructed as: price[t+HORIZON] = price[t] * exp(predicted_return).
#
# Inputs are multivariate: besides the return itself the model sees volume
# change, RSI, MACD and the distance of the price from its moving averages —
# all scale-free, so 1999 and 2026 samples remain comparable.
#
# Note: consecutive training targets overlap by HORIZON-1 days, so they are
# strongly autocorrelated — effective sample size is smaller than it looks.
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM
from keras.callbacks import EarlyStopping

close_s = df['Close']

# --- Technical indicators (all relative/scale-free) ---
# RSI(14), Wilder's smoothing
delta = close_s.diff()
gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
rsi = 100 - 100 / (1 + gain / loss)

# MACD(12, 26, 9), normalized by price so old and new samples are comparable
ema12 = close_s.ewm(span=12, adjust=False).mean()
ema26 = close_s.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
macd_signal = macd.ewm(span=9, adjust=False).mean()

feat = pd.DataFrame(index=df.index)
feat['log_ret'] = np.log(close_s).diff()
feat['vol_chg'] = np.log(df['Volume']).diff()
feat['rsi'] = rsi / 100
feat['macd'] = macd / close_s
feat['macd_hist'] = (macd - macd_signal) / close_s
for ma in ma_days:
    feat[f'ma{ma}_dist'] = close_s / close_s.rolling(ma).mean() - 1

FEATURES = list(feat.columns)

# Target: the cumulative log return over the next HORIZON days
feat['target'] = feat['log_ret'].rolling(HORIZON).sum().shift(-HORIZON)
feat['close'] = close_s
feat = feat.dropna()

# Position of each feature row in the original df, so we can find the DATE
# the prediction is for (HORIZON trading days ahead)
pos = df.index.get_indexer(feat.index)
target_dates = df.index[pos + HORIZON]

n = len(feat)
split = int(np.ceil(n * TRAIN_SPLIT))
print(f"Training on {split} of {n} rows, {len(FEATURES)} features: {FEATURES}")

X_raw = feat[FEATURES].to_numpy()
y_raw = feat['target'].to_numpy()
close_arr = feat['close'].to_numpy()

# Scale features and target. Fit on the TRAINING part only, so no
# information from the validation period leaks into preprocessing.
x_scaler = StandardScaler().fit(X_raw[:split])
y_scaler = StandardScaler().fit(y_raw[:split].reshape(-1, 1))
X = x_scaler.transform(X_raw)
y = y_scaler.transform(y_raw.reshape(-1, 1)).ravel()

# Each sample is WINDOW consecutive days of all features; the target is the
# log return of the day after the window ends
x_train, y_train, x_test = [], [], []
for i in range(WINDOW - 1, n):
    window = X[i - WINDOW + 1:i + 1]
    if i < split:
        x_train.append(window)
        y_train.append(y[i])
    else:
        x_test.append(window)

x_train, y_train = np.array(x_train), np.array(y_train)
x_test = np.array(x_test)
print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

# Build the LSTM model
model = Sequential()
model.add(Input(shape=(WINDOW, len(FEATURES))))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping: keras takes the LAST 10% of the training samples as the
# validation set (chronologically the most recent ones) and stops when the
# validation loss no longer improves, restoring the best weights.
early_stop = EarlyStopping(monitor='val_loss', patience=5,
                           restore_best_weights=True)
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
          validation_split=0.1, callbacks=[early_stop])
print(f"Stopped after {early_stop.stopped_epoch or MAX_EPOCHS} epochs")

# Predict the scaled returns and transform back to real returns
pred_scaled = model.predict(x_test).ravel()
pred_returns = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

# Reconstruct prices: the prediction for HORIZON days after row i is the
# actual close of row i times exp(predicted cumulative return)
base_close = close_arr[split:]                       # close on prediction day
actual = base_close * np.exp(y_raw[split:])          # close HORIZON days later
predictions = base_close * np.exp(pred_returns)

# Baselines. Random walk: the price stays where it is. Drift: the price
# grows at the average HORIZON-day return of the training period — at
# monthly horizons this is the harder baseline to beat.
drift = y_raw[:split].mean()
naive_rw = base_close
naive_drift = base_close * np.exp(drift)

rmse = np.sqrt(np.mean((predictions - actual) ** 2))
rmse_rw = np.sqrt(np.mean((naive_rw - actual) ** 2))
rmse_drift = np.sqrt(np.mean((naive_drift - actual) ** 2))

# Direction hit rate: did the model get the SIGN of the move right?
hit_rate = np.mean(np.sign(pred_returns) == np.sign(y_raw[split:]))
up_share = np.mean(y_raw[split:] > 0)

print(f"Model RMSE: {rmse:.4f}")
print(f"Random-walk RMSE: {rmse_rw:.4f} (price stays flat)")
print(f"Drift RMSE: {rmse_drift:.4f} "
      f"(price grows at the avg training return, {drift:+.4f}/{HORIZON}d)")
print(f"Direction hit rate: {hit_rate:.1%} "
      f"(always-up would score {up_share:.1%})")
if rmse < min(rmse_rw, rmse_drift):
    print("Model beats both naive baselines.")
else:
    print("Model does NOT beat both naive baselines — hard even at "
          "monthly horizons; the hit rate is the more meaningful metric.")


# ---------------------------------------------------------------------------
# 5. Plot the results
# ---------------------------------------------------------------------------
valid_dates = target_dates[split:]
train_series = close_s.loc[:valid_dates[0]]
valid = pd.DataFrame(
    {'Close': actual, 'Predictions': predictions, 'Naive RW': naive_rw,
     'Naive Drift': naive_drift},
    index=valid_dates,
)

plt.figure(figsize=(16, 6))
plt.title(f'{COMPANY_NAME} — LSTM {HORIZON}-Day-Ahead Prediction '
          f'(RMSE {rmse:.2f} vs drift {rmse_drift:.2f}, '
          f'hit rate {hit_rate:.0%})')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_series)
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

# Show the valid and predicted prices
print(valid)

plt.show()
