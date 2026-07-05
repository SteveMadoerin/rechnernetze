"""Central configuration for the stock pipeline.

Every stage reads from a single Config instance so there are no hidden globals
and no magic numbers scattered across the code.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Config:
    # --- Data collection ---
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "GOOG", "MSFT", "AMZN"])
    company_names: List[str] = field(
        default_factory=lambda: ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]
    )
    # Analysis window: last 1 year (set in __post_init__ so it stays "now"-relative).
    start: datetime = None
    end: datetime = None

    # --- LSTM sub-pipeline (single ticker) ---
    model_ticker: str = "AAPL"
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
