"""Production inference service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.etl import ETLPipeline
from src.features.engineering import FeatureEngineeringPipeline
from src.models.ml_models import MLForecaster
from src.utils.config import load_config, PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ForecastPredictor:
    """Load models and serve predictions."""

    def __init__(self, config_path: Optional[Path] = None, model_name: str = "xgboost") -> None:
        self.config = load_config(config_path)
        self.model_name = model_name
        self.etl = ETLPipeline(self.config)
        self.features = FeatureEngineeringPipeline(self.config)
        self.model = None
        self.tft = None
        self._load_model()

    def _load_model(self) -> None:
        model_path = PROJECT_ROOT / "models" / f"{self.model_name}.pkl"
        tft_path = PROJECT_ROOT / "models" / "tft" / "best.ckpt"

        if self.model_name == "tft" and tft_path.exists():
            try:
                from src.models.tft_model import TFTForecaster

                self.tft = TFTForecaster(self.config)
                logger.info("TFT model path configured at %s", tft_path)
            except ImportError as e:
                logger.warning("TFT unavailable (install torch): %s", e)
        elif model_path.exists():
            self.model = MLForecaster.load(model_path)
            logger.info("Loaded model: %s", self.model_name)
        else:
            logger.warning("No trained model found; using heuristic fallback")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.etl.run(df, fit=False)
        return self.features.transform(df)

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 28,
        sku_id: Optional[str] = None,
        dc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self.preprocess(df)
        if sku_id:
            df = df[df["sku_id"] == sku_id]
        if dc_id:
            df = df[df["dc_id"] == dc_id]

        if self.model is not None:
            preds = self.model.predict(df)
        elif self.tft is not None:
            preds = self.tft.predict(df, horizon=horizon)
        else:
            # Fallback: rolling mean
            preds = df.groupby(["sku_id", "dc_id"])["demand"].transform(
                lambda s: s.rolling(7, min_periods=1).mean()
            ).values

        dates = df["date"].tolist() if "date" in df.columns else list(range(len(preds)))
        return {
            "predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds),
            "dates": [str(d) for d in dates[-horizon:]],
            "model": self.model_name,
            "horizon": horizon,
        }

    def batch_predict(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(pd.DataFrame(r.get("history", [])), r.get("horizon", 28)) for r in requests]
