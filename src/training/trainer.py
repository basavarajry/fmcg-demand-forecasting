"""Unified model training orchestrator with MLflow tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd

from src.data.download import DatasetDownloader
from src.data.etl import ETLPipeline
from src.features.engineering import FeatureEngineeringPipeline
from src.models.baseline import BASELINE_MODELS
from src.models.deep_learning import DEEP_MODELS
from src.models.ml_models import ML_MODELS
from src.models.tft_model import TFTForecaster
from src.utils.config import load_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.metrics import ForecastMetrics
from src.utils.seed import set_seed

logger = get_logger(__name__)


class ModelTrainer:
    """Train and compare all forecasting models."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config = load_config(config_path)
        set_seed(self.config.get("training", {}).get("seed", 42))
        self.etl = ETLPipeline(self.config)
        self.features = FeatureEngineeringPipeline(self.config)
        self.results: Dict[str, Dict[str, float]] = {}

        mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "./mlruns"))
        mlflow.set_experiment(self.config.get("project", {}).get("name", "fmcg-forecast"))

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_cfg = self.config["data"]
        downloader = DatasetDownloader(PROJECT_ROOT / data_cfg["raw_dir"])
        df = downloader.load_sales(data_cfg.get("dataset", "synthetic"))
        df = self.etl.run(df, fit=True)
        df = self.features.transform(df)
        ratios = data_cfg.get("train_ratio", 0.7), data_cfg.get("val_ratio", 0.15)
        return self.etl.train_val_test_split(df, train_ratio=ratios[0], val_ratio=ratios[1])

    def train_baseline(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
        kwargs = {
            "group_cols": self.config["data"]["group_cols"],
            "target_col": self.config["data"]["target_col"],
            "date_col": self.config["data"]["date_col"],
        }
        target = test[self.config["data"]["target_col"]].values

        for name, cls in BASELINE_MODELS.items():
            logger.info("Training baseline: %s", name)
            model = cls()
            with mlflow.start_run(run_name=f"baseline_{name}"):
                model.fit(train, val, **kwargs)
                preds = model.predict(test, **kwargs)
                metrics = ForecastMetrics.compute_all(target, preds[: len(target)])
                mlflow.log_metrics(metrics)
                mlflow.set_tag("model_type", "baseline")
                self.results[name] = metrics
        return self.results

    def train_ml(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, tune: bool = False) -> Dict:
        feature_cols = self.features.get_feature_columns(train)
        target = test[self.config["data"]["target_col"]].values
        kwargs = {"target_col": self.config["data"]["target_col"]}

        for name, factory in ML_MODELS.items():
            logger.info("Training ML: %s", name)
            model = factory()
            with mlflow.start_run(run_name=f"ml_{name}"):
                model.fit(train, val, feature_cols=feature_cols, tune=tune, **kwargs)
                preds = model.predict(test, **kwargs)
                metrics = ForecastMetrics.compute_all(target, preds[: len(target)])
                mlflow.log_metrics(metrics)
                model.save(PROJECT_ROOT / "models" / f"{name}.pkl")
                self.results[name] = metrics
        return self.results

    def train_deep(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
        kwargs = {
            "group_cols": self.config["data"]["group_cols"],
            "target_col": self.config["data"]["target_col"],
            "date_col": self.config["data"]["date_col"],
        }
        target = test[self.config["data"]["target_col"]].values

        for name, factory in list(DEEP_MODELS.items())[:2]:  # limit runtime
            logger.info("Training deep: %s", name)
            model = factory()
            with mlflow.start_run(run_name=f"deep_{name}"):
                model.fit(train, val, **kwargs)
                preds = model.predict(test, **kwargs)
                metrics = ForecastMetrics.compute_all(target, preds[: len(target)])
                mlflow.log_metrics(metrics)
                self.results[name] = metrics
        return self.results

    def train_tft(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
        logger.info("Training TFT...")
        model = TFTForecaster(self.config)
        with mlflow.start_run(run_name="tft"):
            model.fit(train, val)
            preds = model.predict(test)
            target = test[self.config["data"]["target_col"]].values
            metrics = ForecastMetrics.compute_all(target, preds[: len(target)])
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model.model, "tft_model")
            model.save_checkpoint(PROJECT_ROOT / "models" / "tft" / "best.ckpt")
            self.results["tft"] = metrics
        return self.results

    def run_full_pipeline(self, phases: Optional[List[str]] = None) -> pd.DataFrame:
        phases = phases or ["baseline", "ml", "deep", "tft"]
        train, val, test = self.load_data()

        if "baseline" in phases:
            self.train_baseline(train, val, test)
        if "ml" in phases:
            self.train_ml(train, val, test, tune=False)
        if "deep" in phases:
            self.train_deep(train, val, test)
        if "tft" in phases:
            self.train_tft(train, val, test)

        comparison = ForecastMetrics.comparison_table(self.results)
        comparison.to_csv(PROJECT_ROOT / "reports" / "model_comparison.csv")
        logger.info("\n%s", comparison)

        if "moving_average" in self.results and "tft" in self.results:
            imp = ForecastMetrics.improvement_pct(
                self.results["moving_average"]["wmape"], self.results["tft"]["wmape"]
            )
            logger.info("TFT WMAPE improvement over MA baseline: %.1f%%", imp)

        return comparison
