"""Stage 3 - Data Analysis (exploratory analysis + plots).

Operates on the 4-stock dataset. Pure read-only analysis: computes stats and
draws figures. Does not train anything.
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


class DataAnalyzer:
    def __init__(self, config: Config) -> None:
        self.config = config

    # --- helpers -------------------------------------------------------
    def _finish(self, name: str) -> None:
        plt.tight_layout()
        if self.config.save_plots:
            plt.savefig(f"{name}.png", bbox_inches="tight")
        if self.config.show_plots:
            plt.show()

    def _tickers(self, data):
        return list(data.keys())

    # --- analyses ------------------------------------------------------
    def summary(self, data: Dict[str, pd.DataFrame]) -> None:
        first = self._tickers(data)[0]
        print(data[first].describe())
        print(data[first].info())

    def plot_closing_price(self, data: Dict[str, pd.DataFrame]) -> None:
        plt.figure(figsize=(15, 10))
        for i, ticker in enumerate(self._tickers(data), 1):
            plt.subplot(2, 2, i)
            data[ticker]["Adj Close"].plot()
            plt.ylabel("Adj Close")
            plt.xlabel(None)
            plt.title(f"Closing Price of {ticker}")
        self._finish("closing_price")

    def plot_volume(self, data: Dict[str, pd.DataFrame]) -> None:
        plt.figure(figsize=(15, 10))
        for i, ticker in enumerate(self._tickers(data), 1):
            plt.subplot(2, 2, i)
            data[ticker]["Volume"].plot()
            plt.ylabel("Volume")
            plt.xlabel(None)
            plt.title(f"Sales Volume for {ticker}")
        self._finish("volume")

    def add_moving_averages(self, data: Dict[str, pd.DataFrame], ma_days=(10, 20, 50)) -> None:
        for ma in ma_days:
            for df in data.values():
                df[f"MA for {ma} days"] = df["Adj Close"].rolling(ma).mean()

        cols = ["Adj Close"] + [f"MA for {ma} days" for ma in ma_days]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        for ax, (ticker, name) in zip(axes.flat, zip(self._tickers(data), self.config.company_names)):
            data[ticker][cols].plot(ax=ax)
            ax.set_title(name)
        self._finish("moving_averages")

    def add_daily_returns(self, data: Dict[str, pd.DataFrame]) -> None:
        for df in data.values():
            df["Daily Return"] = df["Adj Close"].pct_change()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        for ax, (ticker, name) in zip(axes.flat, zip(self._tickers(data), self.config.company_names)):
            data[ticker]["Daily Return"].plot(ax=ax, legend=True, linestyle="--", marker="o")
            ax.set_title(name)
        self._finish("daily_returns")

    def correlation(self, closing_df: pd.DataFrame) -> None:
        tech_rets = closing_df.pct_change()

        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        sns.heatmap(tech_rets.corr(), annot=True, cmap="summer")
        plt.title("Correlation of stock return")
        plt.subplot(2, 2, 2)
        sns.heatmap(closing_df.corr(), annot=True, cmap="summer")
        plt.title("Correlation of stock closing price")
        self._finish("correlation")

    def risk(self, closing_df: pd.DataFrame) -> None:
        rets = closing_df.pct_change().dropna()
        area = np.pi * 20
        plt.figure(figsize=(10, 8))
        plt.scatter(rets.mean(), rets.std(), s=area)
        plt.xlabel("Expected return")
        plt.ylabel("Risk")
        for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
            plt.annotate(
                label, xy=(x, y), xytext=(50, 50), textcoords="offset points",
                ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color="blue", connectionstyle="arc3,rad=-0.3"),
            )
        self._finish("risk")

    def analyze(self, data: Dict[str, pd.DataFrame], closing_df: pd.DataFrame) -> None:
        """Run the full EDA suite."""
        self.summary(data)
        self.plot_closing_price(data)
        self.plot_volume(data)
        self.add_moving_averages(data)
        self.add_daily_returns(data)
        self.correlation(closing_df)
        self.risk(closing_df)
