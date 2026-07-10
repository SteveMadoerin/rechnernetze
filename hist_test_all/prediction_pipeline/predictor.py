"""Stage 5 - Prediction.

Runs the trained model on the test windows, inverse-transforms with the same
scaler used in feature engineering, reports RMSE and plots the result.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

        forecast = None
        if self.config.forecast_days:
            forecast = self._forecast_future(model, dataset, self.config.forecast_days)

        self._plot(dataset, predictions, forecast)

        if forecast is not None:
            print(f"\nForecast for the next {len(forecast)} business days:")
            print(forecast)
        return predictions, rmse

    def _forecast_future(self, model, dataset: Dataset, n_days: int) -> pd.DataFrame:
        """Autoregressive forecast beyond the last known close.

        Feeds the model its own (scaled) predictions one step at a time, so
        uncertainty compounds with the horizon.
        """
        seq = self.config.sequence_length
        window = dataset.scaled[-seq:, :].copy()         # (seq, k)

        scaled_preds = []
        for _ in range(n_days):
            x = window.reshape(1, seq, window.shape[1])
            pred = float(model.predict(x, verbose=0)[0, 0])
            scaled_preds.append(pred)
            # Feed the prediction back as the next close return; exogenous
            # features (volume, range) are unknown for future days, so hold
            # them at their last observed values (persistence assumption).
            next_row = window[-1].copy()
            next_row[0] = pred
            window = np.vstack([window[1:], next_row])

        values = dataset.scaler.inverse_transform(
            np.array(scaled_preds).reshape(-1, 1)
        ).flatten()

        last_price = float(dataset.close_data.values[-1, 0])
        if dataset.use_log_returns:
            prices = last_price * np.exp(np.cumsum(values))
        else:
            prices = values

        last_date = dataset.close_data.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=n_days
        )
        return pd.DataFrame({"Forecast": prices}, index=future_dates)

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

    def _plot(self, dataset: Dataset, predictions, forecast: pd.DataFrame = None) -> None:
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
            legend = ["Train", "Val", "Predictions"]
            if forecast is not None:
                # Anchor the dashed forecast line to the last actual close so
                # it continues the series instead of floating.
                last = dataset.close_data.iloc[[-1]]
                anchor = pd.concat(
                    [last.rename(columns={price_col: "Forecast"}), forecast]
                )
                plt.plot(anchor["Forecast"], "--")
                legend.append(f"Forecast ({len(forecast)} days)")
            plt.legend(legend, loc="lower right")

            if self.config.save_plots:
                plt.savefig("prediction.png", bbox_inches="tight")
            if self.config.show_plots:
                plt.show()

        print(valid)
