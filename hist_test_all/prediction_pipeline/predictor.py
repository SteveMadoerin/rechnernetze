"""Stage 5 - Prediction.

Runs the trained model on the test windows, inverse-transforms with the same
scaler used in feature engineering, reports RMSE and plots the result.
"""

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from features import Dataset


class Predictor:
    def __init__(self, config: Config) -> None:
        self.config = config

    def predict(self, model, dataset: Dataset):
        predictions = model.predict(dataset.x_test)
        predictions = dataset.scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean((predictions - dataset.y_test) ** 2))
        print(f"RMSE: {rmse}")

        self._plot(dataset, predictions)
        return predictions, rmse

    def _plot(self, dataset: Dataset, predictions) -> None:
        data = dataset.close_data
        split = dataset.training_data_len

        train = data[:split]
        valid = data[split:].copy()          # .copy() avoids SettingWithCopyWarning
        valid["Predictions"] = predictions

        plt.figure(figsize=(16, 6))
        plt.title("Model")
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Close Price USD ($)", fontsize=18)
        plt.plot(train["Close"])
        plt.plot(valid[["Close", "Predictions"]])
        plt.legend(["Train", "Val", "Predictions"], loc="lower right")

        if self.config.save_plots:
            plt.savefig("prediction.png", bbox_inches="tight")
        if self.config.show_plots:
            plt.show()

        print(valid)
