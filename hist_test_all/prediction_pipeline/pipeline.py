"""Pipeline orchestrator.

Wires the five stages together and passes data between them explicitly. Two
sub-flows share the same collector/config:
    EDA flow    : collect -> validate -> analyze   (4 stocks)
    Model flow  : collect_single -> features -> train -> predict   (1 stock)

Run:  python pipeline.py
"""

from config import Config
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


if __name__ == "__main__":
    Pipeline().run()
