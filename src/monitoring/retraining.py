"""Automated retraining pipeline with retry logic."""

from __future__ import annotations

from typing import Any, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from src.training.trainer import ModelTrainer
from src.monitoring.drift import DriftMonitor
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrainingPipeline:
    """Orchestrate retraining when drift or accuracy thresholds are breached."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = load_config(config_path)
        self.trainer = ModelTrainer(config_path)
        self.drift_monitor = DriftMonitor(
            threshold=self.config.get("monitoring", {}).get("drift_threshold", 0.1)
        )
        self.wmape_threshold = self.config.get("monitoring", {}).get("wmape_threshold", 0.25)

    def should_retrain(
        self,
        wmape: Optional[float] = None,
        drift_result: Optional[Dict] = None,
        new_data: bool = False,
    ) -> bool:
        if new_data and self.config.get("monitoring", {}).get("retrain_on_new_data", True):
            logger.info("Retrain triggered: new data arrived")
            return True
        if wmape is not None and self.drift_monitor.check_wmape_threshold(wmape, self.wmape_threshold):
            logger.info("Retrain triggered: WMAPE %.2f > threshold %.2f", wmape, self.wmape_threshold)
            return True
        if drift_result and drift_result.get("drift_detected"):
            logger.info("Retrain triggered: data drift detected")
            return True
        return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def run_retraining(self, phases: Optional[list] = None) -> Dict[str, Any]:
        logger.info("Starting automated retraining...")
        comparison = self.trainer.run_full_pipeline(phases=phases or ["ml", "tft"])
        return {"status": "success", "comparison": comparison.to_dict()}
