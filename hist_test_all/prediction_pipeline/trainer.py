"""Stage 4 - Training.

Builds and trains the LSTM model on the prepared Dataset.
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping

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

        callbacks = []
        # Hold out the most recent slice of the training data for validation and
        # stop early once val_loss stops improving (restoring the best weights).
        validation_split = self.config.validation_split
        if validation_split and validation_split > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        self.model.fit(
            dataset.x_train,
            dataset.y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
        )
        return self.model
