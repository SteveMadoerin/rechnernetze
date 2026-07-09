"""DataCamp LSTM tutorial, ported into a single class.

Faithful re-implementation of
    https://www.datacamp.com/tutorial/lstm-python-stock-market

The original tutorial is TensorFlow 1.x (`tf.contrib.rnn`, `tf.nn.dynamic_rnn`),
which no longer exists in modern TF. This ports the *approach* to Keras (already
a dependency here) while keeping the techniques that make the tutorial distinct
from the rest of this pipeline:

  1. Target = mid-price  (High + Low) / 2, not Close/Adj Close.
  2. Windowed MinMaxScaler: fit a separate scaler on each window of the series
     instead of one global (or train-only) fit, because different eras of the
     stock live in very different value ranges.
  3. Exponential-moving-average smoothing of the *training* series (gamma=0.1).
  4. Averaging baselines (standard MA + running EMA) for comparison -- the
     tutorial shows these fail at multi-step prediction.
  5. Multi-step-ahead LSTM prediction: predictions are fed back in as inputs so
     the model forecasts a run of future steps, not just one.

Self-contained: it downloads its own data, so it does not depend on the other
pipeline stages. It reads the ticker from Config if given, else defaults to AAL
(American Airlines), the tutorial's stock.

Run:  python datacamp_lstm.py
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input

try:                                   # optional: reuse the pipeline's Config
    from config import Config
except Exception:                      # allow running fully stand-alone
    Config = None


@dataclass
class DataCampConfig:
    """Hyper-parameters mirroring the tutorial (scaled down to run quickly)."""
    ticker: str = "AAL"                # American Airlines, the tutorial's stock
    train_size: int = 11000            # first N points for training (tutorial)
    smoothing_window: int = 2500       # window for per-window normalisation
    ema_gamma: float = 0.1             # EMA smoothing factor for training data
    num_unrollings: int = 50           # steps unrolled / predicted ahead
    hidden_layers: tuple = (200, 200, 150)   # LSTM stack widths (tutorial)
    dropout: float = 0.2
    batch_size: int = 500
    epochs: int = 5
    show_plots: bool = True


class DataCampLSTM:
    """End-to-end mid-price LSTM following the DataCamp tutorial."""

    def __init__(self, config=None) -> None:
        # Accept either a DataCampConfig, the pipeline Config (for its ticker),
        # or nothing at all.
        if isinstance(config, DataCampConfig):
            self.cfg = config
        elif Config is not None and isinstance(config, Config):
            self.cfg = DataCampConfig(
                ticker=config.model_ticker or "AAL",
                epochs=config.epochs,
            )
        else:
            self.cfg = DataCampConfig()

        self.mid_prices = None
        self.train_data = None
        self.test_data = None
        self.scaler = None
        self.model = None

    # ------------------------------------------------------------------ data
    def load(self) -> np.ndarray:
        """Download OHLC, compute the mid-price series, sort by date."""
        df = yf.download(
            self.cfg.ticker, period="max",
            auto_adjust=False, multi_level_index=False,
        ).sort_index()
        high = df["High"].values
        low = df["Low"].values
        self.mid_prices = ((high + low) / 2.0).astype(float)

        # If the stock is short of 11k points, fall back to a 90/10 split so the
        # class still runs on any ticker.
        n = len(self.mid_prices)
        train_size = min(self.cfg.train_size, int(n * 0.9))
        self.train_data = self.mid_prices[:train_size]
        self.test_data = self.mid_prices[train_size:]
        print(f"{self.cfg.ticker}: {n} mid-prices  "
              f"(train {len(self.train_data)}, test {len(self.test_data)})")
        return self.mid_prices

    # --------------------------------------------------------- normalisation
    def normalize(self) -> None:
        """Windowed MinMax scaling + EMA smoothing of the training series.

        The tutorial fits a fresh scaler on each `smoothing_window` slice so no
        single era dominates, then exponentially smooths the scaled train data.
        The test data is scaled with the *last* fitted scaler.
        """
        window = self.cfg.smoothing_window
        train = self.train_data.reshape(-1, 1).copy()
        self.scaler = MinMaxScaler()

        scaled_train = np.zeros_like(train)
        for lo in range(0, len(train), window):
            hi = min(lo + window, len(train))
            self.scaler.fit(train[lo:hi])
            scaled_train[lo:hi] = self.scaler.transform(train[lo:hi])
        scaled_train = scaled_train.reshape(-1)

        # Exponential moving-average smoothing (tutorial's gamma=0.1).
        ema = 0.0
        gamma = self.cfg.ema_gamma
        for i in range(len(scaled_train)):
            ema = gamma * scaled_train[i] + (1 - gamma) * ema
            scaled_train[i] = ema
        self.train_data = scaled_train

        # Test data uses the last window's scaler.
        scaled_test = self.scaler.transform(
            self.test_data.reshape(-1, 1)).reshape(-1)
        self.test_data = scaled_test

    # ------------------------------------------------------------ baselines
    def averaging_baselines(self) -> None:
        """Standard-average and running-EMA baselines (the tutorial's warm-up).

        Both are one-step predictors; the tutorial uses them to motivate the
        LSTM by showing they cannot extrapolate multiple steps.
        """
        data = np.concatenate([self.train_data, self.test_data])
        window = 100

        # Standard moving average: predict next as mean of the last `window`.
        std_pred, std_err = [], []
        for i in range(window, len(data)):
            std_pred.append(data[i - window:i].mean())
            std_err.append((std_pred[-1] - data[i]) ** 2)
        print(f"Standard-average baseline MSE: {np.mean(std_err):.6f}")

        # Running exponential moving average.
        ema, gamma = 0.0, 0.5
        ema_err = []
        for i in range(1, len(data)):
            ema = gamma * data[i - 1] + (1 - gamma) * ema
            ema_err.append((ema - data[i]) ** 2)
        print(f"Exponential-average baseline MSE: {np.mean(ema_err):.6f}")

    # ---------------------------------------------------------------- windows
    def _sequences(self, series: np.ndarray):
        """Sliding input/target windows of length num_unrollings."""
        n = self.cfg.num_unrollings
        xs, ys = [], []
        for i in range(len(series) - n):
            xs.append(series[i:i + n])
            ys.append(series[i + n])
        x = np.array(xs).reshape(-1, n, 1)
        y = np.array(ys).reshape(-1, 1)
        return x, y

    # ------------------------------------------------------------------ model
    def build(self) -> Sequential:
        n = self.cfg.num_unrollings
        model = Sequential()
        model.add(Input(shape=(n, 1)))
        widths = self.cfg.hidden_layers
        for i, w in enumerate(widths):
            model.add(LSTM(w, return_sequences=(i < len(widths) - 1)))
            if self.cfg.dropout:
                model.add(Dropout(self.cfg.dropout))
        model.add(Dense(1))            # linear regression head (tutorial's w, b)
        model.compile(optimizer="adam", loss="mean_squared_error")
        self.model = model
        return model

    def train(self) -> Sequential:
        if self.model is None:
            self.build()
        x_train, y_train = self._sequences(self.train_data)
        self.model.fit(
            x_train, y_train,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
        )
        return self.model

    # ------------------------------------------------------------- inference
    def predict_multistep(self, n_steps: int = None) -> np.ndarray:
        """Seed with the last training window, then feed predictions back in.

        This is the tutorial's key move: each prediction becomes the next input,
        so the model forecasts `n_steps` ahead rather than one step.
        """
        n = self.cfg.num_unrollings
        n_steps = n_steps or n
        window = list(self.train_data[-n:])
        preds = []
        for _ in range(n_steps):
            x = np.array(window[-n:]).reshape(1, n, 1)
            yhat = float(self.model.predict(x, verbose=0)[0, 0])
            preds.append(yhat)
            window.append(yhat)
        return np.array(preds)

    def evaluate_onestep(self):
        """One-step MSE over the test set (comparable to the baselines)."""
        x_test, y_test = self._sequences(
            np.concatenate([self.train_data[-self.cfg.num_unrollings:],
                            self.test_data]))
        pred = self.model.predict(x_test, verbose=0)
        mse = float(np.mean((pred - y_test) ** 2))
        print(f"LSTM one-step test MSE (normalised): {mse:.6f}")
        return pred, y_test, mse

    # ---------------------------------------------------------------- plot
    def plot(self, preds: np.ndarray) -> None:
        if not self.cfg.show_plots:
            return
        plt.figure(figsize=(16, 6))
        plt.title(f"{self.cfg.ticker} - DataCamp LSTM (normalised mid-price)")
        history = np.concatenate([self.train_data, self.test_data])
        plt.plot(range(len(history)), history, color="b", label="True")
        start = len(self.train_data)
        plt.plot(range(start, start + len(preds)), preds,
                 color="r", label="Multi-step prediction")
        plt.xlabel("Time step")
        plt.ylabel("Normalised mid-price")
        plt.legend()
        plt.show()

    # ---------------------------------------------------------------- run all
    def run(self):
        self.load()
        self.normalize()
        self.averaging_baselines()
        self.train()
        self.evaluate_onestep()
        preds = self.predict_multistep()
        self.plot(preds)
        return preds


if __name__ == "__main__":
    DataCampLSTM().run()
