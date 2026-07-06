"""Feature engineering for the LSTM sub-pipeline.

Scales the 'Close' series and turns it into sliding look-back windows. Holds the
fitted scaler so Prediction can inverse-transform with the *same* scaler.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import Config


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    scaled_data: np.ndarray
    training_data_len: int
    close_data: pd.DataFrame   # single-column 'Close' DataFrame, for plotting


class FeatureEngineer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def prepare(self, df: pd.DataFrame) -> Dataset:
        seq = self.config.sequence_length

        close_data = df.filter(["Close"])
        dataset = close_data.values
        training_data_len = int(np.ceil(len(dataset) * self.config.train_split))

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # --- training windows ---
        train_data = scaled_data[0:training_data_len, :]
        x_train, y_train = [], []
        for i in range(seq, len(train_data)):
            x_train.append(train_data[i - seq:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # --- test windows ---
        test_data = scaled_data[training_data_len - seq:, :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(seq, len(test_data)):
            x_test.append(test_data[i - seq:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return Dataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            scaler=scaler,
            scaled_data=scaled_data,
            training_data_len=training_data_len,
            close_data=close_data,
        )
