"""Forecast evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    """Compute and compare forecast accuracy metrics."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        denom = np.maximum(np.abs(y_true), epsilon)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    @staticmethod
    def wmape(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        if weights is None:
            weights = np.abs(y_true)
        weights = np.maximum(weights, 1e-8)
        return float(np.sum(np.abs(y_true - y_pred)) / np.sum(weights) * 100)

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
        return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).flatten()
        y_pred = np.asarray(y_pred, dtype=float).flatten()
        return {
            "mae": cls.mae(y_true, y_pred),
            "rmse": cls.rmse(y_true, y_pred),
            "mape": cls.mape(y_true, y_pred),
            "wmape": cls.wmape(y_true, y_pred, weights),
            "smape": cls.smape(y_true, y_pred),
        }

    @classmethod
    def comparison_table(
        cls,
        results: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """Build model comparison DataFrame from nested metric dicts."""
        return pd.DataFrame(results).T.sort_values("wmape")

    @classmethod
    def improvement_pct(cls, baseline_wmape: float, model_wmape: float) -> float:
        if baseline_wmape <= 0:
            return 0.0
        return float((baseline_wmape - model_wmape) / baseline_wmape * 100)
