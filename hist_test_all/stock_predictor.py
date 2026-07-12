"""LSTM stock forecaster wrapped in a single class.

Combines the structure of thepythoncode.com's TF2/Keras tutorial
(load_data / create_model / profit evaluation / future prediction) with the
methodology worked out in analyseSly.py:

- full-history CSV cache per ticker
- scale-free multivariate features (returns, volume change, RSI, MACD,
  distance from moving averages), so 1999 and 2026 samples are comparable
- target = cumulative log return over the next `horizon` days
  (direct multi-step strategy, no error accumulation)
- scalers fit on the training period only (no leakage)
- honest evaluation: RMSE vs random-walk and drift baselines, direction
  hit rate, and a non-overlapping long-only trading backtest vs buy & hold

GPU: TensorFlow >= 2.11 is CPU-only on native Windows; run through WSL2:
  wsl -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/Users/steve/PycharmProjects/rechnernetze/hist_test_all/run_wsl_gpu.sh > /tmp/run_gpu_hta.sh && bash /tmp/run_gpu_hta.sh stock_predictor.py"
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping

sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


class StockPredictor:
    """End-to-end pipeline: load -> plot -> features -> train -> evaluate."""

    MA_DAYS = [10, 20, 50]

    def __init__(self, ticker, company_name=None, window=60, horizon=21,
                 train_split=0.95, max_epochs=50, batch_size=32,
                 lstm_units=(128, 64), dropout=0.2, bidirectional=False,
                 data_dir=None):
        self.ticker = ticker
        self.company_name = company_name or ticker
        self.window = window
        self.horizon = horizon
        self.train_split = train_split
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.data_dir,
                                     f"{self.ticker}_history.csv")

        self.df = None          # raw OHLCV history
        self.feat = None        # feature/target frame
        self.model = None
        self.results = None     # evaluation metrics dict

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    def load_data(self):
        """Load the full price history from CSV cache, else download."""
        if os.path.exists(self.csv_path):
            print(f"Loading {self.ticker} data from {self.csv_path}")
            self.df = pd.read_csv(self.csv_path, index_col=0,
                                  parse_dates=True)
        else:
            print(f"Downloading full {self.ticker} history ...")
            self.df = yf.download(
                self.ticker,
                period='max',
                auto_adjust=False,
                multi_level_index=False,
            )
            self.df.to_csv(self.csv_path)
            print(f"Saved {len(self.df)} rows to {self.csv_path}")
        return self.df

    def plot_data(self):
        """Overview figure: price, volume, moving averages, daily returns."""
        df = self.df.copy()
        for ma in self.MA_DAYS:
            df[f'MA {ma}'] = df['Adj Close'].rolling(ma).mean()
        df['Daily Return'] = df['Adj Close'].pct_change()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        df['Adj Close'].plot(ax=axes[0, 0],
                             title=f'Closing Price of {self.company_name}')
        df['Volume'].plot(ax=axes[0, 1],
                          title=f'Sales Volume for {self.company_name}')
        df[['Adj Close'] + [f'MA {ma}' for ma in self.MA_DAYS]].plot(
            ax=axes[1, 0], title=f'Moving Averages of {self.company_name}')
        df['Daily Return'].hist(bins=50, ax=axes[1, 1])
        axes[1, 1].set_title(f'Daily Return Distribution '
                             f'of {self.company_name}')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. Features
    # ------------------------------------------------------------------
    def prepare_features(self):
        """Build scale-free features and the horizon-return target."""
        close = self.df['Close']

        # RSI(14), Wilder's smoothing
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss)

        # MACD(12, 26, 9), normalized by price
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        feat = pd.DataFrame(index=self.df.index)
        feat['log_ret'] = np.log(close).diff()
        feat['vol_chg'] = np.log(self.df['Volume']).diff()
        feat['rsi'] = rsi / 100
        feat['macd'] = macd / close
        feat['macd_hist'] = (macd - signal) / close
        for ma in self.MA_DAYS:
            feat[f'ma{ma}_dist'] = close / close.rolling(ma).mean() - 1

        self.feature_names = list(feat.columns)

        # Target: cumulative log return over the next `horizon` days.
        # Consecutive targets overlap by horizon-1 days (autocorrelated).
        feat['target'] = (feat['log_ret'].rolling(self.horizon).sum()
                          .shift(-self.horizon))
        feat['close'] = close
        self.feat = feat.dropna()

        # Date each prediction refers to (horizon trading days ahead)
        pos = self.df.index.get_indexer(self.feat.index)
        self.target_dates = self.df.index[pos + self.horizon]

        n = len(self.feat)
        self.split = int(np.ceil(n * self.train_split))
        print(f"Training on {self.split} of {n} rows, "
              f"{len(self.feature_names)} features: {self.feature_names}")

        X_raw = self.feat[self.feature_names].to_numpy()
        y_raw = self.feat['target'].to_numpy()

        # Fit scalers on the TRAINING part only (no leakage)
        self.x_scaler = StandardScaler().fit(X_raw[:self.split])
        self.y_scaler = StandardScaler().fit(
            y_raw[:self.split].reshape(-1, 1))
        self.X = self.x_scaler.transform(X_raw)
        self.y = self.y_scaler.transform(y_raw.reshape(-1, 1)).ravel()
        self.y_raw = y_raw
        self.close_arr = self.feat['close'].to_numpy()

        # Sliding windows: sample at row i covers rows [i-window+1, i]
        x_train, y_train, x_test = [], [], []
        for i in range(self.window - 1, n):
            w = self.X[i - self.window + 1:i + 1]
            if i < self.split:
                x_train.append(w)
                y_train.append(self.y[i])
            else:
                x_test.append(w)
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.x_test = np.array(x_test)
        print(f"x_train shape: {self.x_train.shape}, "
              f"x_test shape: {self.x_test.shape}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    def create_model(self):
        """LSTM stack with dropout, optionally bidirectional."""
        model = Sequential()
        model.add(Input(shape=(self.window, len(self.feature_names))))
        for i, units in enumerate(self.lstm_units):
            layer = LSTM(units,
                         return_sequences=(i < len(self.lstm_units) - 1))
            if self.bidirectional:
                layer = Bidirectional(layer)
            model.add(layer)
            model.add(Dropout(self.dropout))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def train(self):
        """Train with early stopping on the most recent 10% of train data."""
        if self.model is None:
            self.create_model()
        early = EarlyStopping(monitor='val_loss', patience=5,
                              restore_best_weights=True)
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size, epochs=self.max_epochs,
                       validation_split=0.1, callbacks=[early])
        print(f"Stopped after {early.stopped_epoch or self.max_epochs} "
              f"epochs")

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    def evaluate(self):
        """RMSE vs naive baselines, hit rate, and a trading backtest."""
        pred_scaled = self.model.predict(self.x_test).ravel()
        self.pred_returns = self.y_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)).ravel()

        actual_ret = self.y_raw[self.split:]
        base_close = self.close_arr[self.split:]
        actual = base_close * np.exp(actual_ret)
        predictions = base_close * np.exp(self.pred_returns)

        drift = self.y_raw[:self.split].mean()
        naive_rw = base_close
        naive_drift = base_close * np.exp(drift)

        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        rmse_rw = np.sqrt(np.mean((naive_rw - actual) ** 2))
        rmse_drift = np.sqrt(np.mean((naive_drift - actual) ** 2))
        hit_rate = np.mean(np.sign(self.pred_returns) == np.sign(actual_ret))
        up_share = np.mean(actual_ret > 0)

        # Long-only trading backtest on NON-overlapping horizon windows:
        # every `horizon` days, go long iff the model predicts a gain.
        # Compare against buy & hold over the same windows.
        idx = np.arange(0, len(actual_ret), self.horizon)
        take = self.pred_returns[idx] > 0
        strategy_ret = np.exp(np.sum(actual_ret[idx][take])) - 1
        buyhold_ret = np.exp(np.sum(actual_ret[idx])) - 1

        self.valid = pd.DataFrame(
            {'Close': actual, 'Predictions': predictions,
             'Naive RW': naive_rw, 'Naive Drift': naive_drift},
            index=self.target_dates[self.split:],
        )
        self.results = {
            'rmse': rmse, 'rmse_rw': rmse_rw, 'rmse_drift': rmse_drift,
            'hit_rate': hit_rate, 'up_share': up_share,
            'strategy_return': strategy_ret, 'buyhold_return': buyhold_ret,
            'n_windows': len(idx), 'n_trades': int(take.sum()),
        }

        print(f"Model RMSE: {rmse:.4f}")
        print(f"Random-walk RMSE: {rmse_rw:.4f} (price stays flat)")
        print(f"Drift RMSE: {rmse_drift:.4f} "
              f"(avg training return, {drift:+.4f}/{self.horizon}d)")
        print(f"Direction hit rate: {hit_rate:.1%} "
              f"(always-up would score {up_share:.1%})")
        print(f"Trading backtest ({self.results['n_windows']} "
              f"non-overlapping {self.horizon}d windows, long when the "
              f"model predicts a gain, {self.results['n_trades']} trades):")
        print(f"  strategy return: {strategy_ret:+.1%}")
        print(f"  buy & hold:      {buyhold_ret:+.1%}")
        if rmse < min(rmse_rw, rmse_drift):
            print("Model beats both naive baselines.")
        else:
            print("Model does NOT beat both naive baselines — hard even at "
                  "monthly horizons; hit rate and backtest matter more.")
        return self.results

    def plot_results(self):
        train_series = self.feat['close'].iloc[:self.split]
        r = self.results
        fig = plt.figure(figsize=(16, 6))
        plt.title(f"{self.company_name} — LSTM {self.horizon}-Day-Ahead "
                  f"Prediction (RMSE {r['rmse']:.2f} vs drift "
                  f"{r['rmse_drift']:.2f}, hit rate {r['hit_rate']:.0%})")
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train_series)
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        return fig

    # ------------------------------------------------------------------
    # 5. Future prediction
    # ------------------------------------------------------------------
    def predict_future(self):
        """Predict the price `horizon` trading days after the last close."""
        last_window = self.X[-self.window:].reshape(
            1, self.window, len(self.feature_names))
        pred_scaled = self.model.predict(last_window, verbose=0).ravel()
        pred_ret = float(self.y_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)).ravel()[0])
        last_close = float(self.close_arr[-1])
        future_price = last_close * np.exp(pred_ret)
        future_date = self.feat.index[-1] + pd.tseries.offsets.BDay(
            self.horizon)
        print(f"Last close {self.feat.index[-1].date()}: {last_close:.2f}")
        print(f"Predicted close ~{future_date.date()} "
              f"({self.horizon} trading days ahead): {future_price:.2f} "
              f"({pred_ret:+.2%})")
        return future_price

    # ------------------------------------------------------------------
    def run(self, show_plots=True):
        self.load_data()
        print(self.df.tail(10))
        self.plot_data()
        self.prepare_features()
        self.train()
        self.evaluate()
        self.plot_results()
        self.predict_future()
        print(self.valid)
        if show_plots:
            plt.show()
        return self.results


if __name__ == '__main__':
    predictor = StockPredictor('NVDA', company_name='NVIDIA')
    predictor.run()
