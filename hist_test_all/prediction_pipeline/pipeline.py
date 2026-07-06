"""Pipeline orchestrator.

Wires the five stages together and passes data between them explicitly. Two
sub-flows share the same collector/config:
    EDA flow    : collect -> validate -> analyze   (4 stocks)
    Model flow  : collect_single -> features -> train -> predict   (1 stock)

Run:  python pipeline.py
"""

from config import Config, MAX_STOCKS, AVAILABLE_STOCKS, DEFAULT_TICKERS
from collector import DataCollector
from validator import DataValidator, ValidationError
from analyzer import DataAnalyzer
from features import FeatureEngineer
from trainer import ModelTrainer
from predictor import Predictor


class Pipeline:
    def __init__(self, config: Config = None) -> None:
        self.config = config or Config()
        self.collector = DataCollector(self.config)
        self.validator = DataValidator(self.config)
        self.analyzer = DataAnalyzer(self.config)
        self.features = FeatureEngineer(self.config)
        self.trainer = ModelTrainer(self.config)
        self.predictor = Predictor(self.config)

    def run_eda(self) -> None:
        """Stages 1-3 on the multi-stock dataset."""
        data = self.collector.collect()

        warnings = self.validator.validate(data)
        for w in warnings:
            print(f"[warning] {w}")

        closing_df = self.collector.collect_closing_prices()
        self.analyzer.analyze(data, closing_df)

    def run_model(self):
        """Stages 4-5 (with feature engineering) on a single ticker."""
        df = self.collector.collect_single()

        # Validate the single-ticker frame too.
        try:
            self.validator.validate({self.config.model_ticker: df})
        except ValidationError as e:
            raise SystemExit(f"Cannot train: {e}")

        dataset = self.features.prepare(df)
        model = self.trainer.train(dataset)
        return self.predictor.predict(model, dataset)

    def run(self) -> None:
        self.run_eda()
        self.run_model()


def prompt_stocks() -> Config:
    """Let the user pick which stocks to analyse.

    Accepts either tickers (e.g. "AAPL, NVDA") or menu numbers (e.g. "1 5").
    Empty input uses the default selection. Re-prompts on invalid input.
    """
    menu = list(AVAILABLE_STOCKS.items())
    print("Available stocks:")
    for i, (ticker, name) in enumerate(menu, 1):
        print(f"  {i}. {ticker:<6} {name}")
    print(f"Pick up to {MAX_STOCKS} by ticker or number, comma/space separated.")

    while True:
        raw = input(f"Your choice [default {DEFAULT_TICKERS}]: ").strip()
        if raw == "":
            return Config.from_tickers(DEFAULT_TICKERS)

        tokens = [t for t in raw.replace(",", " ").split() if t]
        tickers = []
        for tok in tokens:
            if tok.isdigit():                       # menu number -> ticker
                idx = int(tok)
                if not 1 <= idx <= len(menu):
                    tickers = None
                    print(f"Number out of range: {tok}")
                    break
                tickers.append(menu[idx - 1][0])
            else:                                   # a ticker symbol
                tickers.append(tok.upper())
        if tickers is None:
            continue

        try:
            config = Config.from_tickers(tickers)
        except ValueError as e:
            print(f"Invalid selection: {e}")
            continue
        return config


def prompt_prediction_target(config: Config) -> None:
    """Ask which of the selected stocks the LSTM should predict.

    Sets config.model_ticker in place. Empty input keeps the first selected.
    """
    tickers = config.tickers
    if len(tickers) == 1:                       # nothing to choose
        config.model_ticker = tickers[0]
        return

    options = ", ".join(f"{i+1}={t}" for i, t in enumerate(tickers))
    while True:
        raw = input(f"Which stock to predict? ({options}) [default {tickers[0]}]: ").strip()
        if raw == "":
            config.model_ticker = tickers[0]
            return
        if raw.isdigit() and 1 <= int(raw) <= len(tickers):
            config.model_ticker = tickers[int(raw) - 1]
            return
        if raw.upper() in tickers:
            config.model_ticker = raw.upper()
            return
        print(f"Please pick one of {tickers} (or its number).")


if __name__ == "__main__":
    config = prompt_stocks()
    prompt_prediction_target(config)
    print(f"Selected stocks: {config.tickers}  |  predicting: {config.model_ticker}")
    Pipeline(config).run()
