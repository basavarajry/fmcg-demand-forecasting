"""FastAPI production inference service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.auth import verify_token
from src.api.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    RetrainRequest,
    RetrainResponse,
)
from src.inference.predictor import ForecastPredictor
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_predictor: ForecastPredictor | None = None
_metrics_cache: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    settings = get_settings()
    _predictor = ForecastPredictor(model_name="xgboost")
    logger.info("Forecast API started (env=%s)", settings.app_env)
    yield
    logger.info("Forecast API shutdown")


app = FastAPI(
    title="FMCG Demand Forecasting API",
    description="AI-Powered Multi-Horizon Demand Forecasting for FMCG Supply Chains",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=_predictor is not None,
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast(
    request: ForecastRequest,
    _: str = Depends(verify_token),
):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([r.model_dump() for r in request.history])
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _predictor.predict(df, horizon=request.horizon, sku_id=request.sku_id, dc_id=request.dc_id),
    )
    return ForecastResponse(**result)


@app.post("/batch_forecast", response_model=BatchForecastResponse, tags=["Forecasting"])
async def batch_forecast(
    request: BatchForecastRequest,
    _: str = Depends(verify_token),
):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for req in request.requests:
        df = pd.DataFrame([r.model_dump() for r in req.history])
        result = _predictor.predict(df, horizon=req.horizon, sku_id=req.sku_id, dc_id=req.dc_id)
        results.append(ForecastResponse(**result))
    return BatchForecastResponse(results=results)


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def metrics(_: str = Depends(verify_token)):
    return MetricsResponse(
        wmape=_metrics_cache.get("wmape"),
        mae=_metrics_cache.get("mae"),
        last_evaluated=_metrics_cache.get("last_evaluated"),
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(_: str = Depends(verify_token)):
    settings = get_settings()
    features = []
    if _predictor and _predictor.model and hasattr(_predictor.model, "feature_cols"):
        features = _predictor.model.feature_cols
    return ModelInfoResponse(
        model_name=_predictor.model_name if _predictor else "none",
        model_type="ml" if _predictor and _predictor.model else "fallback",
        features=features,
        horizon=settings.forecast_horizon,
        encoder_length=settings.encoder_length,
    )


@app.post("/retrain", response_model=RetrainResponse, tags=["MLOps"])
async def retrain(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_token),
):
    def _run_retrain():
        from src.monitoring.retraining import RetrainingPipeline

        pipeline = RetrainingPipeline()
        pipeline.run_retraining(phases=request.phases)

    if request.force:
        background_tasks.add_task(_run_retrain)
        return RetrainResponse(status="started", message="Retraining job queued", job_id=datetime.utcnow().isoformat())
    return RetrainResponse(status="accepted", message="Submit with force=true to start retraining")
