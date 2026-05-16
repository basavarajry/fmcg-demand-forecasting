"""Configuration loading utilities."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_secret_key: str = Field(default="dev-secret", alias="API_SECRET_KEY")
    api_access_token: str = Field(default="dev-token", alias="API_ACCESS_TOKEN")

    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    model_dir: Path = Field(default=PROJECT_ROOT / "models", alias="MODEL_DIR")
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")

    forecast_horizon: int = Field(default=28, alias="FORECAST_HORIZON")
    encoder_length: int = Field(default=90, alias="ENCODER_LENGTH")
    wmape_retrain_threshold: float = Field(default=0.25, alias="WMAPE_RETRAIN_THRESHOLD")

    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_bucket: str = Field(default="fmcg-forecast-models", alias="S3_BUCKET")

    mlflow_experiment_name: str = Field(
        default="fmcg-demand-forecasting", alias="MLFLOW_EXPERIMENT_NAME"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
