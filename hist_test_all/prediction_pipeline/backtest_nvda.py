"""Backtest: train the pipeline LSTM on NVDA data up to 2026-07-06, then
forecast forward and compare against the actual closes since the cutoff."""

import sys

import pandas as pd
import yfinance as yf

from config import Config
from features import FeatureEngineer
from trainer import ModelTrainer
from predictor import Predictor

CUTOFF = "2026-07-06"

full = yf.download("NVDA", period="max", auto_adjust=False, multi_level_index=False)
train_df = full.loc[:CUTOFF]
actual_after = full.loc[full.index > train_df.index[-1], "Adj Close"]

print(f"Full history : {full.index[0].date()} .. {full.index[-1].date()} ({len(full)} rows)")
print(f"Training cut : ends {train_df.index[-1].date()} ({len(train_df)} rows)")
print(f"Held-out days: {[str(d) for d in actual_after.index.date]}")

n_days = len(actual_after)
if n_days == 0:
    sys.exit("No actual data after the cutoff to compare against.")

config = Config.from_tickers(
    ["NVDA"],
    model_type="pipeline",
    forecast_days=n_days,
    epochs=30,
    run_eda=False,
    plots=[],
    show_plots=False,
)

dataset = FeatureEngineer(config).prepare(train_df)
model = ModelTrainer(config).train(dataset)
forecast = Predictor(config)._forecast_future(model, dataset, n_days)

cmp = pd.DataFrame(
    {
        "Forecast": forecast["Forecast"].values[:n_days],
        "Actual": actual_after.values,
    },
    index=actual_after.index,
)
cmp["Error $"] = cmp["Forecast"] - cmp["Actual"]
cmp["Error %"] = cmp["Error $"] / cmp["Actual"] * 100

last_close = float(train_df["Adj Close"].iloc[-1])
print(f"\nLast known close ({train_df.index[-1].date()}): {last_close:.2f}")
print("\nForecast vs actual:")
print(cmp.round(2).to_string())
print(f"\nMean absolute error: {cmp['Error $'].abs().mean():.2f} USD "
      f"({cmp['Error %'].abs().mean():.2f}%)")
