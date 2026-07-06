"""Stage 1 - Data Collection.

Downloads raw OHLCV data from Yahoo Finance. Returns plain dicts/DataFrames;
no globals, no plotting, no analysis.
"""

from typing import Dict

import pandas as pd
import yfinance as yf

from config import Config


class DataCollector:
    def __init__(self, config: Config) -> None:
        self.config = config

    def collect(self) -> Dict[str, pd.DataFrame]:
        """Download one DataFrame per ticker, tagged with its company name."""
        data: Dict[str, pd.DataFrame] = {}
        for ticker, name in zip(self.config.tickers, self.config.company_names):
            df = yf.download(
                ticker,
                start=self.config.start,
                end=self.config.end,
                auto_adjust=False,          # keep a separate "Adj Close" column
                multi_level_index=False,    # flat columns for a single ticker
            )
            df["company_name"] = name
            data[ticker] = df
        return data

    def collect_closing_prices(self) -> pd.DataFrame:
        """All tickers' 'Adj Close' in one DataFrame (used for correlation)."""
        return yf.download(
            self.config.tickers,
            start=self.config.start,
            end=self.config.end,
            auto_adjust=False,
        )["Adj Close"]

    def collect_single(self, ticker: str = None, start: str = None) -> pd.DataFrame:
        """Full price history for one ticker, used by the LSTM sub-pipeline.

        If no start date is given (config.model_start is None), fetches the
        stock's entire available history (period='max') so each company starts
        from its own earliest available date.
        """
        ticker = ticker or self.config.model_ticker
        start = start if start is not None else self.config.model_start
        common = dict(auto_adjust=False, multi_level_index=False)
        if start is None:
            # period and start/end are mutually exclusive in yfinance.
            return yf.download(ticker, period="max", **common)
        return yf.download(ticker, start=start, end=self.config.end, **common)
