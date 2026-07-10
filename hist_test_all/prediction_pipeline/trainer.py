"""Stage 4 - Training.

Builds and trains the LSTM model on the prepared Dataset.
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout
from keras.callbacks import EarlyStopping

from config import Config
from features import Dataset


class ModelTrainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None

    def build(self, sequence_length: int, n_features: int = 1) -> Sequential:
        # Smaller stack + dropout than the original 128/64: less capacity to
        # memorise, which pulls val_loss (and RMSE) down on this 1-D signal.
        dropout = self.config.dropout
        model = Sequential()
        model.add(Input(shape=(sequence_length, n_features)))
        model.add(LSTM(64, return_sequences=True))
        if dropout:
            model.add(Dropout(dropout))
        model.add(LSTM(32, return_sequences=False))
        if dropout:
            model.add(Dropout(dropout))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        self.model = model
        return model

    def train(self, dataset: Dataset) -> Sequential:
        if self.model is None:
            self.build(self.config.sequence_length, dataset.x_train.shape[2])

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
