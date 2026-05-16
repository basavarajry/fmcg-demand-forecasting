"""SHAP explainability for ML models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """Generate global and local SHAP explanations."""

    def __init__(self, model, feature_cols: List[str]) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.explainer = None
        self.shap_values = None

    def fit_explainer(self, X: pd.DataFrame, max_samples: int = 500) -> None:
        if shap is None:
            logger.warning("SHAP not installed")
            return
        X_sample = X[self.feature_cols].fillna(0)
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(max_samples, random_state=42)

        if hasattr(self.model, "predict"):
            self.explainer = shap.TreeExplainer(self.model.model if hasattr(self.model, "model") else self.model)
            self.shap_values = self.explainer.shap_values(X_sample)
        logger.info("SHAP explainer fitted on %d samples", len(X_sample))

    def global_importance(self) -> pd.DataFrame:
        if self.shap_values is None:
            return pd.DataFrame()
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return pd.DataFrame({"feature": self.feature_cols, "shap_importance": mean_abs}).sort_values(
            "shap_importance", ascending=False
        )

    def local_explanation(self, idx: int = 0) -> Dict[str, float]:
        if self.shap_values is None:
            return {}
        return dict(zip(self.feature_cols, self.shap_values[idx]))

    def save_summary_plot(self, output_path: Path) -> None:
        if shap is None or self.shap_values is None:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, feature_names=self.feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary saved to %s", output_path)

    def generate_report(self, X: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.fit_explainer(X)
        global_imp = self.global_importance()
        global_imp.to_csv(output_dir / "shap_global_importance.csv", index=False)
        self.save_summary_plot(output_dir / "shap_summary.png")
        return {"global_importance": global_imp.to_dict(), "status": "complete"}
