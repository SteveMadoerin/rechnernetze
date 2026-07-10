"""Walk-forward backtest for the pipeline LSTM.

Instead of judging the model on a single cutoff date (see backtest_nvda.py),
this trains a fresh model at N_CUTOFFS points spaced STEP business days apart,
forecasts HORIZON days from each, and aggregates the errors. Every variant is
compared against the naive baseline (last close carried forward) — a forecast
only has value if it beats that.

Run:  python backtest_walkforward.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import yfinance as yf
import keras

from config import Config
from features import FeatureEngineer
from trainer import ModelTrainer
from predictor import Predictor

TICKER = "NVDA"
HORIZON = 5        # business days forecast from each cutoff
N_CUTOFFS = 8      # how many cutoff dates to evaluate
STEP = 10          # business days between cutoffs
EPOCHS = 30        # max epochs; early stopping usually ends sooner
VARIANTS = {"multivariate": True, "univariate": False}


def run_variant(full: pd.DataFrame, cutoff_positions, multivariate: bool):
    """Per-cutoff forecasts; returns abs % errors, shape (n_cutoffs, HORIZON)."""
    errors = []
    for pos in cutoff_positions:
        keras.backend.clear_session()        # fresh graph per model
        train_df = full.iloc[: pos + 1]
        actual = full["Adj Close"].iloc[pos + 1 : pos + 1 + HORIZON].values

        config = Config.from_tickers(
            [TICKER],
            multivariate=multivariate,
            epochs=EPOCHS,
            forecast_days=HORIZON,
            plots=[],
            show_plots=False,
        )
        dataset = FeatureEngineer(config).prepare(train_df)
        model = ModelTrainer(config).train(dataset)
        forecast = Predictor(config)._forecast_future(model, dataset, HORIZON)

        pct_err = np.abs(forecast["Forecast"].values - actual) / actual * 100
        errors.append(pct_err)
        print(f"  cutoff {full.index[pos].date()}: "
              f"MAPE {pct_err.mean():.2f}%  (day1 {pct_err[0]:.2f}%)")
    return np.array(errors)


def main():
    full = yf.download(TICKER, period="max",
                       auto_adjust=False, multi_level_index=False)

    # Newest cutoff leaves HORIZON actual days after it; older ones step back.
    last_pos = len(full) - 1 - HORIZON
    cutoff_positions = [last_pos - i * STEP for i in range(N_CUTOFFS)][::-1]
    print(f"{TICKER}: {len(full)} rows, cutoffs "
          f"{full.index[cutoff_positions[0]].date()} .. "
          f"{full.index[cutoff_positions[-1]].date()} "
          f"({N_CUTOFFS} x {HORIZON}-day forecasts)\n")

    # Naive baseline: last known close carried forward over the horizon.
    naive = []
    closes = full["Adj Close"].values
    for pos in cutoff_positions:
        actual = closes[pos + 1 : pos + 1 + HORIZON]
        naive.append(np.abs(closes[pos] - actual) / actual * 100)
    results = {"naive (last close)": np.array(naive)}

    for name, multivariate in VARIANTS.items():
        print(f"[{name}]")
        results[name] = run_variant(full, cutoff_positions, multivariate)
        print()

    day_cols = [f"day{i+1}" for i in range(HORIZON)]
    table = pd.DataFrame(
        {name: errs.mean(axis=0) for name, errs in results.items()},
        index=day_cols,
    ).T
    table["overall"] = [errs.mean() for errs in results.values()]

    print("Mean absolute % error per horizon day "
          f"(avg over {N_CUTOFFS} cutoffs):")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
