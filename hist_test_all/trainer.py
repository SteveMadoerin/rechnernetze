"""Stage 4 - Training.

Builds and trains the LSTM model on the prepared Dataset.
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Input

from config import Config
from features import Dataset


class ModelTrainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None

    def build(self, sequence_length: int) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(sequence_length, 1)))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        self.model = model
        return model

    def train(self, dataset: Dataset) -> Sequential:
        if self.model is None:
            self.build(self.config.sequence_length)
        self.model.fit(
            dataset.x_train,
            dataset.y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
        )
        return self.model
