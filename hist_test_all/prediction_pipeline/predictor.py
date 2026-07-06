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
        # RMSE as a percentage of the mean actual price -> comparable across
        # stocks trading at very different price levels.
        mean_price = float(np.mean(dataset.y_test))
        rmse_pct = rmse / mean_price * 100 if mean_price else float("nan")
        print(f"RMSE: {rmse:.4f}  ({rmse_pct:.2f}% of mean price {mean_price:.2f})")

        self._plot(dataset, predictions)
        return predictions, rmse

    def _plot(self, dataset: Dataset, predictions) -> None:
        data = dataset.close_data
        split = dataset.training_data_len
        price_col = dataset.price_col

        train = data[:split]
        valid = data[split:].copy()          # .copy() avoids SettingWithCopyWarning
        valid["Predictions"] = predictions

        # The prediction table is always printed; the chart is opt-in.
        if "prediction" in self.config.plots:
            plt.figure(figsize=(16, 6))
            plt.title("Model")
            plt.xlabel("Date", fontsize=18)
            plt.ylabel(f"{price_col} Price USD ($)", fontsize=18)
            plt.plot(train[price_col])
            plt.plot(valid[[price_col, "Predictions"]])
            plt.legend(["Train", "Val", "Predictions"], loc="lower right")

            if self.config.save_plots:
                plt.savefig("prediction.png", bbox_inches="tight")
            if self.config.show_plots:
                plt.show()

        print(valid)
