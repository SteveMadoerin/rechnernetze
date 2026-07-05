"""Stage 2 - Data Validation.

Checks the raw data is usable before any analysis or training. Raises
ValidationError on hard failures; returns warnings for soft issues.
"""

from typing import Dict, List

import pandas as pd

from config import Config


class ValidationError(Exception):
    """Raised when data is unfit to proceed through the pipeline."""


class DataValidator:
    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    def __init__(self, config: Config) -> None:
        self.config = config

    def validate(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate every ticker's DataFrame. Returns a list of soft warnings.

        Raises ValidationError if any DataFrame is unusable.
        """
        warnings: List[str] = []
        for ticker, df in data.items():
            warnings.extend(self._validate_one(ticker, df))
        return warnings

    def _validate_one(self, ticker: str, df: pd.DataFrame) -> List[str]:
        warnings: List[str] = []

        if df is None or df.empty:
            raise ValidationError(f"{ticker}: no data returned")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValidationError(f"{ticker}: missing columns {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValidationError(f"{ticker}: index is not a DatetimeIndex")

        # Soft issues -> warnings, not fatal.
        na_counts = df[self.REQUIRED_COLUMNS].isna().sum()
        for col, n in na_counts.items():
            if n > 0:
                warnings.append(f"{ticker}: {n} NaN values in '{col}'")

        if (df["Volume"] < 0).any():
            warnings.append(f"{ticker}: negative volume values present")

        if df.index.duplicated().any():
            warnings.append(f"{ticker}: duplicate dates in index")

        return warnings
