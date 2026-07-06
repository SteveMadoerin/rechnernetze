"""Central configuration for the stock pipeline.

Every stage reads from a single Config instance so there are no hidden globals
and no magic numbers scattered across the code.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

# The universe of stocks you can pick from (ticker -> display name). Add more
# here and they become selectable automatically. Max 4 can be *selected* at once
# because the analysis plots use a 2x2 grid.
MAX_STOCKS = 4
AVAILABLE_STOCKS = {
    "AAPL": "APPLE",
    "GOOG": "GOOGLE",
    "MSFT": "MICROSOFT",
    "AMZN": "AMAZON",
    "NVDA": "NVIDIA",
    "META": "META",
    "TSLA": "TESLA",
    "NFLX": "NETFLIX",
}
# Default selection = first 4 of the universe.
DEFAULT_TICKERS = list(AVAILABLE_STOCKS)[:MAX_STOCKS]
DEFAULT_COMPANY_NAMES = [AVAILABLE_STOCKS[t] for t in DEFAULT_TICKERS]


@dataclass
class Config:
    # --- Data collection ---
    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    company_names: List[str] = field(
        default_factory=lambda: list(DEFAULT_COMPANY_NAMES)
    )
    # Analysis window: last 1 year (set in __post_init__ so it stays "now"-relative).
    start: datetime = None
    end: datetime = None

    # --- LSTM sub-pipeline (single ticker) ---
    # If left None, defaults to the first selected ticker.
    model_ticker: str = None
    model_start: str = "2012-01-01"
    sequence_length: int = 60          # look-back window fed into the LSTM
    train_split: float = 0.95          # fraction of data used for training
    batch_size: int = 1
    epochs: int = 1

    # --- Output / behaviour ---
    show_plots: bool = True            # call plt.show(); set False for headless runs
    save_plots: bool = False           # save figures to disk instead of/along with showing

    def __post_init__(self) -> None:
        if self.end is None:
            self.end = datetime.now()
        if self.start is None:
            self.start = datetime(self.end.year - 1, self.end.month, self.end.day)
        if len(self.tickers) != len(self.company_names):
            raise ValueError("tickers and company_names must have the same length")
        if not 1 <= len(self.tickers) <= MAX_STOCKS:
            raise ValueError(f"number of stocks must be between 1 and {MAX_STOCKS}")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("duplicate tickers in selection")
        # Prediction runs on the first selected stock unless told otherwise.
        if self.model_ticker is None:
            self.model_ticker = self.tickers[0]

    @classmethod
    def from_count(cls, count: int, **kwargs) -> "Config":
        """Build a Config using the first `count` default stocks (max 4)."""
        if not 1 <= count <= MAX_STOCKS:
            raise ValueError(f"count must be between 1 and {MAX_STOCKS}, got {count}")
        return cls.from_tickers(DEFAULT_TICKERS[:count], **kwargs)

    @classmethod
    def from_tickers(cls, tickers: List[str], **kwargs) -> "Config":
        """Build a Config from an explicit ticker list (each must be known, max 4)."""
        tickers = [t.strip().upper() for t in tickers]
        unknown = [t for t in tickers if t not in AVAILABLE_STOCKS]
        if unknown:
            raise ValueError(
                f"unknown ticker(s) {unknown}; choose from {list(AVAILABLE_STOCKS)}"
            )
        return cls(
            tickers=tickers,
            company_names=[AVAILABLE_STOCKS[t] for t in tickers],
            **kwargs,
        )
