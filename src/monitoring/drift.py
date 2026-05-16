"""Data and concept drift detection with Evidently AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftMonitor:
    """Monitor data drift and forecast accuracy degradation."""

    def __init__(self, threshold: float = 0.1, report_dir: Path = Path("reports/evidently")) -> None:
        self.threshold = threshold
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def detect_data_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
            report.save_html(str(self.report_dir / "data_drift_report.html"))

            result = report.as_dict()
            drift_detected = False
            metrics = result.get("metrics", [])
            for m in metrics:
                if "drift" in str(m).lower():
                    drift_detected = True
                    break

            return {"drift_detected": drift_detected, "report_path": str(self.report_dir / "data_drift_report.html")}
        except ImportError:
            logger.warning("Evidently not available; using statistical fallback")
            return self._statistical_drift(reference, current)

    def _statistical_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = reference.select_dtypes(include=["number"]).columns
        drift_cols = []
        for col in numeric_cols:
            if col not in current.columns:
                continue
            ref_mean, cur_mean = reference[col].mean(), current[col].mean()
            ref_std = reference[col].std() + 1e-8
            z_score = abs(cur_mean - ref_mean) / ref_std
            if z_score > 2.0:
                drift_cols.append(col)

        drift_detected = len(drift_cols) / max(len(numeric_cols), 1) > self.threshold
        return {"drift_detected": drift_detected, "drifted_columns": drift_cols}

    def check_wmape_threshold(self, wmape: float, threshold: float) -> bool:
        return wmape > threshold
