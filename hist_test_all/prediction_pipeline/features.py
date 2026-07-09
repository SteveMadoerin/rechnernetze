"""Feature engineering for the LSTM sub-pipeline.

Turns the price series into sliding look-back windows and scales them. Holds the
fitted scaler so Prediction can inverse-transform with the *same* scaler.

Two target modes (config.use_log_returns):
  * log-returns (default): predict log(P[t]/P[t-1]). Stationary, so recent test
    values stay in the training range -> no extrapolation problem.
  * price level: predict the (scaled) Adj Close directly. Simple but drifts out
    of range for a rising stock.

Uses 'Adj Close' (split/dividend-adjusted) so long histories fetched with
period='max' don't contain artificial split discontinuities.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import Config

PRICE_COL = "Adj Close"


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray             # actual prices for the test period (both modes)
    scaler: MinMaxScaler
    training_data_len: int         # index into the price series where test begins
    close_data: pd.DataFrame       # single-column price DataFrame, for plotting
    price_col: str                 # name of the price column ("Adj Close")
    use_log_returns: bool          # how to turn model output back into prices


class FeatureEngineer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def prepare(self, df: pd.DataFrame) -> Dataset:
        close_data = df.filter([PRICE_COL])
        prices = close_data.values                       # shape (n, 1)
        training_data_len = int(np.ceil(len(prices) * self.config.train_split))

        if self.config.use_log_returns:
            return self._prepare_returns(close_data, prices, training_data_len)
        return self._prepare_price(close_data, prices, training_data_len)

    # ------------------------------------------------------------------
    def _windows(self, scaled, lo, hi):
        """Windows whose TARGET index t runs over [lo, hi)."""
        seq = self.config.sequence_length
        xs, ys = [], []
        for t in range(lo, hi):
            xs.append(scaled[t - seq:t, 0])
            ys.append(scaled[t, 0])
        x = np.array(xs)
        y = np.array(ys)
        if len(x):
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        return x, y

    # ------------------------------------------------------------------
    def _prepare_price(self, close_data, prices, split) -> Dataset:
        seq = self.config.sequence_length

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(prices[:split])                       # train-only fit (no leakage)
        scaled = scaler.transform(prices)

        x_train, y_train = self._windows(scaled, seq, split)
        x_test, _ = self._windows(scaled, split, len(prices))
        y_test = prices[split:, :]                       # actual prices

        return Dataset(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            scaler=scaler, training_data_len=split, close_data=close_data,
            price_col=PRICE_COL, use_log_returns=False,
        )

    # ------------------------------------------------------------------
    def _prepare_returns(self, close_data, prices, split) -> Dataset:
        seq = self.config.sequence_length
        flat = prices.flatten()

        # returns[k] = log(P[k+1] / P[k]); target price index t maps to return t-1.
        returns = np.diff(np.log(flat)).reshape(-1, 1)   # length n-1
        # Training returns are those whose target price index < split, i.e. r[0..split-2].
        train_ret_len = split - 1

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(returns[:train_ret_len])              # train-only fit
        scaled = scaler.transform(returns)

        # Work in return-index space (j = t - 1).
        x_train, y_train = self._windows(scaled, seq, train_ret_len)
        x_test, _ = self._windows(scaled, train_ret_len, len(returns))
        y_test = prices[split:, :]                        # actual prices for RMSE

        return Dataset(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            scaler=scaler, training_data_len=split, close_data=close_data,
            price_col=PRICE_COL, use_log_returns=True,
        )
