"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HistoryRecord(BaseModel):
    date: str
    sku_id: str
    dc_id: str
    demand: float
    promo_flag: Optional[int] = 0
    discount_pct: Optional[float] = 0
    inventory_ratio: Optional[float] = 1.0
    competitor_price_idx: Optional[float] = 1.0
    web_trend: Optional[float] = 50.0


class ForecastRequest(BaseModel):
    history: List[HistoryRecord] = Field(..., min_length=7)
    horizon: int = Field(default=28, ge=1, le=90)
    sku_id: Optional[str] = None
    dc_id: Optional[str] = None
    model_name: Optional[str] = "xgboost"


class ForecastResponse(BaseModel):
    predictions: List[float]
    dates: List[str]
    model: str
    horizon: int
    quantiles: Optional[Dict[str, List[float]]] = None


class BatchForecastRequest(BaseModel):
    requests: List[ForecastRequest] = Field(..., max_length=1000)


class BatchForecastResponse(BaseModel):
    results: List[ForecastResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    wmape: Optional[float] = None
    mae: Optional[float] = None
    last_evaluated: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    features: List[str]
    horizon: int
    encoder_length: int


class RetrainRequest(BaseModel):
    phases: Optional[List[str]] = Field(default=["ml", "tft"])
    force: bool = False


class RetrainResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None
