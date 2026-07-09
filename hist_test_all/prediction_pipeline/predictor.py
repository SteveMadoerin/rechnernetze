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
        raw = model.predict(dataset.x_test)
        predictions = self._to_prices(dataset, raw)

        rmse = np.sqrt(np.mean((predictions - dataset.y_test) ** 2))
        # RMSE as a percentage of the mean actual price -> comparable across
        # stocks trading at very different price levels.
        mean_price = float(np.mean(dataset.y_test))
        rmse_pct = rmse / mean_price * 100 if mean_price else float("nan")
        print(f"RMSE: {rmse:.4f}  ({rmse_pct:.2f}% of mean price {mean_price:.2f})")

        self._plot(dataset, predictions)
        return predictions, rmse

    def _to_prices(self, dataset: Dataset, raw) -> np.ndarray:
        """Turn raw model output into predicted prices (shape (m, 1))."""
        values = dataset.scaler.inverse_transform(raw)
        if not dataset.use_log_returns:
            return values                                # already prices

        # Return mode: values are one-step log-returns. Reconstruct each price
        # from the *actual* previous close (honest one-step-ahead forecast).
        log_rets = values.flatten()
        prices = dataset.close_data.values.flatten()
        split = dataset.training_data_len
        prev = prices[split - 1: split - 1 + len(log_rets)]
        return (prev * np.exp(log_rets)).reshape(-1, 1)

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
