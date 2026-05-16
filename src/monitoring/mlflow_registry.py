"""MLflow model registry utilities."""

from __future__ import annotations

from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Register and promote models in MLflow."""

    def __init__(self, tracking_uri: str = "./mlruns") -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str, artifact_path: str = "model") -> str:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        logger.info("Registered model %s version %s", model_name, result.version)
        return result.version

    def promote_to_production(self, model_name: str, version: str) -> None:
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )
        logger.info("Promoted %s v%s to Production", model_name, version)

    def load_production_model(self, model_name: str):
        return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
