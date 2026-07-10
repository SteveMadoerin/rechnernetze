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
    scaled: np.ndarray             # full scaled series, seed for future forecasts


class FeatureEngineer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def prepare(self, df: pd.DataFrame) -> Dataset:
        close_data = df.filter([PRICE_COL])
        prices = close_data.values                       # shape (n, 1)
        training_data_len = int(np.ceil(len(prices) * self.config.train_split))

        if self.config.use_log_returns:
            return self._prepare_returns(df, close_data, prices, training_data_len)
        return self._prepare_price(close_data, prices, training_data_len)

    # ------------------------------------------------------------------
    def _extra_features(self, df: pd.DataFrame) -> np.ndarray:
        """Exogenous features aligned to return index j (= info from day j+1).

        Column 1: log volume change  log(V[j+1] / V[j])
        Column 2: intraday range     (High - Low) / Close  on day j+1
        Both are stationary, like the target return, so MinMax scaling on the
        training slice generalises to the test period.
        """
        vol = np.clip(df["Volume"].values.astype(float), 1.0, None)
        vol_change = np.log(vol[1:] / vol[:-1])
        day_range = ((df["High"] - df["Low"]) / df["Close"]).values[1:]
        return np.column_stack([vol_change, day_range])

    # ------------------------------------------------------------------
    def _windows(self, scaled, lo, hi):
        """Windows whose TARGET index t runs over [lo, hi).

        Inputs are all feature columns; the target is always column 0 (the
        scaled close price or return). Returns x with shape (m, seq, k).
        """
        seq = self.config.sequence_length
        xs, ys = [], []
        for t in range(lo, hi):
            xs.append(scaled[t - seq:t, :])
            ys.append(scaled[t, 0])
        x = np.array(xs)
        y = np.array(ys)
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
            price_col=PRICE_COL, use_log_returns=False, scaled=scaled,
        )

    # ------------------------------------------------------------------
    def _prepare_returns(self, df, close_data, prices, split) -> Dataset:
        seq = self.config.sequence_length
        flat = prices.flatten()

        # returns[k] = log(P[k+1] / P[k]); target price index t maps to return t-1.
        returns = np.diff(np.log(flat)).reshape(-1, 1)   # length n-1
        # Training returns are those whose target price index < split, i.e. r[0..split-2].
        train_ret_len = split - 1

        # The target scaler covers only column 0 so Prediction can inverse-
        # transform model output without touching the exogenous columns.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(returns[:train_ret_len])              # train-only fit
        scaled = scaler.transform(returns)

        needed = {"Volume", "High", "Low", "Close"}
        if self.config.multivariate and needed.issubset(df.columns):
            extra = self._extra_features(df)             # same length n-1
            extra_scaler = MinMaxScaler(feature_range=(0, 1))
            extra_scaler.fit(extra[:train_ret_len])      # train-only fit
            scaled = np.hstack([scaled, extra_scaler.transform(extra)])

        # Work in return-index space (j = t - 1).
        x_train, y_train = self._windows(scaled, seq, train_ret_len)
        x_test, _ = self._windows(scaled, train_ret_len, len(returns))
        y_test = prices[split:, :]                        # actual prices for RMSE

        return Dataset(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            scaler=scaler, training_data_len=split, close_data=close_data,
            price_col=PRICE_COL, use_log_returns=True, scaled=scaled,
        )
