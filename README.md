# AI-Powered Multi-Horizon Demand Forecasting System for FMCG Supply Chains

Enterprise-grade demand forecasting platform for FMCG/retail supply chains. Forecasts SKU-level demand across distribution centers using **Temporal Fusion Transformer (TFT)**, gradient boosting, statistical baselines, and full MLOps.

## Features

| Phase | Capability |
|-------|------------|
| **Data** | ETL, validation, M5/Favorita/UCI download, synthetic data generator |
| **Baselines** | Moving Average, ARIMA, SARIMA, ETS with seasonal decomposition |
| **Features** | Lags, rolling stats, Fourier, holidays, promotions, inventory, web trends |
| **ML** | XGBoost, LightGBM, Random Forest, CatBoost + Optuna tuning |
| **Deep Learning** | LSTM, GRU, Seq2Seq, Attention-LSTM (multi-horizon) |
| **TFT** | Quantile forecasting, static/dynamic features, attention interpretability |
| **Explainability** | SHAP, attention heatmaps, forecast driver analysis |
| **MLOps** | MLflow, Evidently drift, automated retraining, model registry |
| **API** | FastAPI with `/forecast`, `/batch_forecast`, `/retrain`, `/metrics` |
| **Deploy** | Docker, AWS EC2/S3/Lambda/CloudWatch, GitHub Actions CI/CD |
| **Dashboard** | Streamlit with forecasts, SHAP, drift monitoring |

## Quick Start

```bash
# 1. Clone and setup
cd fmcg-demand-forecasting
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env

# 2. Generate data
python scripts/download_data.py --dataset synthetic

# 3. Train models
python scripts/train.py --phases baseline ml tft

# 4. Start API
uvicorn src.api.main:app --reload --port 8000

# 5. Dashboard
streamlit run dashboards/app.py
```

## Project Structure

```
fmcg-demand-forecasting/
├── data/                 # raw, processed, external
├── configs/              # YAML config + data schema
├── src/
│   ├── data/             # ETL, download, synthetic, validation
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # Baseline, ML, DL, TFT
│   ├── training/         # Trainer, cross-validation
│   ├── inference/        # Production predictor
│   ├── explainability/   # SHAP, attention viz
│   ├── monitoring/       # Drift, retraining, MLflow registry
│   └── api/              # FastAPI service
├── dashboards/           # Streamlit app
├── deployment/aws/       # EC2, Lambda, CloudWatch
├── scripts/              # train, download, reports
├── tests/
└── notebooks/
```

## API Usage

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{
    "history": [{"date":"2024-01-01","sku_id":"SKU_0001","dc_id":"DC_00","demand":100}],
    "horizon": 28
  }'
```

Swagger docs: http://localhost:8000/docs

## Docker

```bash
docker compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# MLflow: http://localhost:5000
```

## AWS Deployment

1. Sync models: `aws s3 sync models/ s3://your-bucket/models/`
2. Run `bash deployment/aws/deploy_ec2.sh`
3. Configure Lambda (`deployment/aws/lambda_retrain.py`) for S3/CloudWatch triggers
4. Set CloudWatch alarms per `deployment/aws/cloudwatch_alarms.json`

## Configuration

Edit `configs/config.yaml` for horizons, TFT hyperparameters, drift thresholds, and dataset selection.

## Metrics

- MAE, RMSE, MAPE, **WMAPE**, SMAPE
- Target: **15–20% WMAPE reduction** vs statistical baselines

## Documentation

- [Architecture](docs/architecture.md)
- [Data Flow](docs/data_flow.md)
- [Data Dictionary](configs/data_schema.yaml)

## License

MIT
