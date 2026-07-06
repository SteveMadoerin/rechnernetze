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

    def _grid(self, n: int):
        """Return (fig, axes_list) sized to n stocks (1->1x1, 2->1x2, 3/4->2x2).

        axes_list is trimmed to exactly n; any leftover cells are hidden.
        """
        cols = 1 if n == 1 else 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10), squeeze=False)
        flat = list(axes.flat)
        for ax in flat[n:]:          # hide unused cells (e.g. 3 stocks in a 2x2)
            ax.set_visible(False)
        return fig, flat[:n]

    # --- analyses ------------------------------------------------------
    def summary(self, data: Dict[str, pd.DataFrame]) -> None:
        first = self._tickers(data)[0]
        print(data[first].describe())
        print(data[first].info())

    def plot_closing_price(self, data: Dict[str, pd.DataFrame]) -> None:
        tickers = self._tickers(data)
        fig, axes = self._grid(len(tickers))
        for ax, ticker in zip(axes, tickers):
            data[ticker]["Adj Close"].plot(ax=ax)
            ax.set_ylabel("Adj Close")
            ax.set_xlabel(None)
            ax.set_title(f"Closing Price of {ticker}")
        self._finish("closing_price")

    def plot_volume(self, data: Dict[str, pd.DataFrame]) -> None:
        tickers = self._tickers(data)
        fig, axes = self._grid(len(tickers))
        for ax, ticker in zip(axes, tickers):
            data[ticker]["Volume"].plot(ax=ax)
            ax.set_ylabel("Volume")
            ax.set_xlabel(None)
            ax.set_title(f"Sales Volume for {ticker}")
        self._finish("volume")

    def add_moving_averages(self, data: Dict[str, pd.DataFrame], ma_days=(10, 20, 50)) -> None:
        for ma in ma_days:
            for df in data.values():
                df[f"MA for {ma} days"] = df["Adj Close"].rolling(ma).mean()

        cols = ["Adj Close"] + [f"MA for {ma} days" for ma in ma_days]
        tickers = self._tickers(data)
        fig, axes = self._grid(len(tickers))
        for ax, ticker, name in zip(axes, tickers, self.config.company_names):
            data[ticker][cols].plot(ax=ax)
            ax.set_title(name)
        self._finish("moving_averages")

    def add_daily_returns(self, data: Dict[str, pd.DataFrame]) -> None:
        for df in data.values():
            df["Daily Return"] = df["Adj Close"].pct_change()

        tickers = self._tickers(data)
        fig, axes = self._grid(len(tickers))
        for ax, ticker, name in zip(axes, tickers, self.config.company_names):
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
        # Correlation and risk compare stocks against each other -> need >= 2.
        if len(self._tickers(data)) >= 2:
            self.correlation(closing_df)
            self.risk(closing_df)
        else:
            print("[info] Only one stock selected - skipping correlation and risk plots.")
