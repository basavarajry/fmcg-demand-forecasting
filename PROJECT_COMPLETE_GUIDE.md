# FMCG Demand Forecasting — Complete Project Guide (A to Z)

**Project name:** AI-Powered Multi-Horizon Demand Forecasting System for FMCG Supply Chains  
**Location:** `c:\Users\basav\OneDrive\Desktop\fmcg-demand-forecasting`  
**Status:** Complete end-to-end platform (data → models → API → dashboard → deployment)

---

## Table of Contents

1. [Why You Built This Project](#1-why-you-built-this-project)
2. [Real-World Use & Business Value](#2-real-world-use--business-value)
3. [What Problem It Solves](#3-what-problem-it-solves)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Technologies Used & Why](#5-technologies-used--why)
6. [Datasets Explained](#6-datasets-explained)
7. [All 10 Phases Explained](#7-all-10-phases-explained)
8. [Dashboard Explained](#8-dashboard-explained)
9. [Docker Explained](#9-docker-explained)
10. [AWS & Deployment Explained](#10-aws--deployment-explained)
11. [MLOps & Monitoring](#11-mlops--monitoring)
12. [How Data Flows Through the System](#12-how-data-flows-through-the-system)
13. [Every File in the Project](#13-every-file-in-the-project)
14. [What You Can Say in Interviews / Resume](#14-what-you-can-say-in-interviews--resume)
15. [Quick Commands Reference](#15-quick-commands-reference)

---

## 1. Why You Built This Project

You built this project to simulate how **large retailers and FMCG companies** (Walmart, Amazon, P&G, Unilever) forecast product demand across **hundreds of SKUs** and **dozens of warehouses**.

Traditional methods (moving averages, simple Excel forecasts) fail when:

- Demand spikes during promotions
- Holidays change buying patterns
- Competitor pricing affects sales
- Each SKU behaves differently per warehouse

This project shows you can build a **production-grade ML system** — not just a notebook — that:

- Ingests and cleans real supply-chain data
- Trains many model types (simple → advanced)
- Uses **Temporal Fusion Transformer (TFT)** — industry-standard deep learning for demand forecasting
- Explains *why* a forecast was made (SHAP, attention)
- Serves predictions via **REST API**
- Monitors drift and retrains automatically
- Deploys with **Docker** and **AWS**

**In one sentence:** You built an enterprise AI platform that predicts how much of each product each warehouse will need in the next 7–28 days, so companies can reduce stockouts and overstock.

---

## 2. Real-World Use & Business Value

| Use case | How this project helps |
|----------|-------------------------|
| **Inventory planning** | Forecast demand → order right quantity from suppliers |
| **Warehouse allocation** | Per-DC forecasts → ship stock to where it's needed |
| **Promotion planning** | Promo flags as features → predict uplift during sales |
| **Reduce stockouts** | Better accuracy on high-demand SKUs |
| **Reduce waste** | Don't overstock slow-moving items |
| **Executive dashboards** | Streamlit UI for planners and analysts |
| **Automated systems** | API integrates with ERP/WMS tools |

**Target metric:** Reduce **WMAPE** (Weighted Mean Absolute Percentage Error) by **15–20%** compared to simple statistical baselines — directly translates to millions saved at scale.

---

## 3. What Problem It Solves

**Business scenario modeled:**

A retailer with **500+ SKUs** and **50 distribution centers** suffers from:

- Frequent **stockouts** on popular items (lost sales)
- **Overstocking** slow movers (waste, storage cost)
- Poor accuracy from moving averages
- Cannot use promotions, holidays, pricing, trends in forecasts

**Your system's answer:**

1. Combine sales history + promotions + holidays + inventory + external signals
2. Train models from simple (ARIMA) to advanced (TFT)
3. Pick the best model per use case
4. Serve forecasts via API and dashboard
5. Retrain when accuracy drops or data drifts

---

## 4. High-Level Architecture

```
[Data Sources] → [ETL + Features] → [Models] → [MLflow] → [API / Dashboard]
                      ↓                  ↓
               [Validation]      [SHAP / Attention]
                      ↓
               [Evidently Drift] → [Auto Retrain] → [AWS S3 / EC2]
```

**Three layers:**

1. **Data layer** — download, clean, validate, engineer features  
2. **ML layer** — train baselines, ML, deep learning, TFT  
3. **Serving layer** — FastAPI, Streamlit, Docker, AWS  

---

## 5. Technologies Used & Why

### Core language & data

| Technology | Why used |
|------------|----------|
| **Python** | Standard for ML, data science, and APIs |
| **Pandas** | Tabular time-series manipulation |
| **NumPy** | Fast numerical operations |
| **PyArrow / Parquet** | Efficient storage for large datasets |

### Statistical & ML models

| Technology | Why used |
|------------|----------|
| **Statsmodels** | ARIMA, SARIMA, ETS — classic baselines every forecaster compares against |
| **pmdarima** | Auto-ARIMA hyperparameter search |
| **Scikit-learn** | Preprocessing, Random Forest, metrics |
| **XGBoost** | Strong tabular forecaster with feature importance |
| **LightGBM** | Fast gradient boosting on large data |
| **CatBoost** | Handles categoricals well (SKU, DC, brand) |
| **Optuna** | Automated hyperparameter tuning |

### Deep learning

| Technology | Why used |
|------------|----------|
| **PyTorch** | Industry-standard deep learning framework |
| **PyTorch Lightning** | Clean training loops, checkpoints, GPU, early stopping |
| **pytorch-forecasting** | Ready-made **Temporal Fusion Transformer** for time series |
| **TFT** | Handles static features (SKU category), known future (promos), unknown past (sales) — used by Amazon, Google research |

### Explainability

| Technology | Why used |
|------------|----------|
| **SHAP** | Shows which features drove each prediction (trust for business users) |
| **Attention visualization** | TFT-specific: which past days the model "looked at" |

### MLOps

| Technology | Why used |
|------------|----------|
| **MLflow** | Track experiments, log metrics, version models |
| **Evidently AI** | Detect when new data differs from training data (drift) |
| **TensorBoard** | Training loss curves for deep models |

### API & deployment

| Technology | Why used |
|------------|----------|
| **FastAPI** | Fast, async, auto Swagger docs, Pydantic validation |
| **Uvicorn** | ASGI server for FastAPI |
| **Docker** | Same environment on laptop, server, cloud |
| **docker-compose** | Run API + dashboard + MLflow together |
| **AWS S3** | Store models and data |
| **AWS EC2** | GPU server for training/serving |
| **AWS Lambda** | Trigger retraining on events |
| **AWS CloudWatch** | Logs and alarms |
| **GitHub Actions** | CI/CD: test on every push |

### Dashboard

| Technology | Why used |
|------------|----------|
| **Streamlit** | Quick interactive web UI without writing HTML/JS |
| **Plotly** | Interactive charts (zoom, hover) |

### Config & quality

| Technology | Why used |
|------------|----------|
| **YAML** | Human-readable config (horizons, paths, thresholds) |
| **Pydantic** | Validate API requests/responses |
| **pytest** | Automated tests |
| **python-dotenv** | Secrets and env-specific settings |

---

## 6. Datasets Explained

### Supported datasets (in `src/data/download.py`)

| Dataset | Source | What it represents | When to use |
|---------|--------|-------------------|-------------|
| **synthetic** | Generated by your code | Fake but realistic FMCG sales | **Default** — works offline, no download |
| **m5** | Walmart M5 Competition (Zenodo) | 42,000+ SKU retail demand | Real-world scale benchmarking |
| **favorita** | Kaggle (Favorita grocery) | Ecuador grocery sales | Another retail benchmark |
| **uci** | UCI Online Retail | Transaction-level sales | Smaller academic dataset |

### Synthetic dataset (what you likely have now)

**Files in `data/raw/`:**

- `synthetic_sales.parquet` — main sales table  
- `sku_metadata.parquet` — product info (category, brand)  
- `dc_metadata.parquet` — warehouse info (region, capacity)  

**Columns in sales data:**

| Column | Meaning |
|--------|---------|
| `date` | Day of sale |
| `sku_id` | Product ID (e.g. SKU_0001) |
| `dc_id` | Distribution center (e.g. DC_00) |
| `demand` | Units sold (**target** to predict) |
| `promo_flag` | Was item on promotion? |
| `discount_pct` | Discount percentage |
| `inventory_ratio` | Stock level signal |
| `competitor_price_idx` | Relative competitor pricing |
| `web_trend` | Search/trend interest |
| `is_holiday` | Holiday indicator |
| `category`, `brand`, `region` | Static attributes for TFT |

**Why synthetic?** Real M5 is ~5GB; synthetic lets you develop and demo without Kaggle accounts or long downloads.

### Data dictionary

Full schema: `configs/data_schema.yaml`

---

## 7. All 10 Phases Explained

| Phase | What was built | File(s) |
|-------|----------------|---------|
| **1 – Baselines** | Moving Average, ARIMA, SARIMA, ETS | `src/models/baseline.py` |
| **2 – Features** | Lags, rolling stats, Fourier, holidays | `src/features/engineering.py` |
| **3 – ML** | XGBoost, LightGBM, RF, CatBoost + Optuna | `src/models/ml_models.py` |
| **4 – Deep Learning** | LSTM, GRU, Seq2Seq, Attention | `src/models/deep_learning.py` |
| **5 – TFT** | Temporal Fusion Transformer (main model) | `src/models/tft_model.py` |
| **6 – Explainability** | SHAP + attention heatmaps | `src/explainability/` |
| **7 – MLOps** | MLflow, drift, retraining | `src/monitoring/` |
| **8 – API** | FastAPI REST service | `src/api/` |
| **9 – Docker/AWS** | Containers + cloud scripts | `Dockerfile`, `deployment/aws/` |
| **10 – Dashboard** | Streamlit UI | `dashboards/app.py` |

### Evaluation metrics (all models compared on these)

- **MAE** — average absolute error in units  
- **RMSE** — penalizes large errors more  
- **MAPE** — percentage error  
- **WMAPE** — weighted by volume (important for retail; high-volume SKUs matter more)  
- **SMAPE** — symmetric percentage error  

Results saved to: `reports/model_comparison.csv`

---

## 8. Dashboard Explained

**File:** `dashboards/app.py`  
**Run:** `streamlit run dashboards/app.py`  
**URL:** http://localhost:8501

### Five tabs

| Tab | Purpose | What you see |
|-----|---------|--------------|
| **Forecasts** | Main use | Select SKU + DC + horizon; line chart of actual vs MA-7 vs future forecast |
| **SKU Analytics** | Product view | Bar chart of total demand per SKU; find top sellers |
| **Warehouse** | DC view | Pie chart of demand share per distribution center |
| **Model Performance** | ML ops view | Table/chart of WMAPE by model (after training); drift report link |
| **Explainability** | Trust & debugging | SHAP feature importance bar chart; TFT attention heatmap (if generated) |

### How dashboard gets data

- **Demo mode:** Generates small synthetic sample on load (`SyntheticDataGenerator`)  
- **After training:** Reads `reports/model_comparison.csv`, SHAP CSV, drift HTML from disk  

### Why Streamlit?

- Planners and managers need **visual** tools, not curl commands  
- Built in Python — same language as ML code  
- Fast to build; good for portfolios and demos  

---

## 9. Docker Explained

### What is Docker?

Docker packages your app + Python + dependencies into a **container** — a lightweight virtual box that runs the same everywhere.

### Why use it in this project?

| Reason | Explanation |
|--------|-------------|
| **"Works on my machine"** | Teammates and cloud servers get identical environment |
| **Production** | Companies deploy APIs as containers (Kubernetes, ECS) |
| **Multi-service** | API + dashboard + MLflow run together via compose |
| **Portfolio** | Shows you understand DevOps, not only notebooks |

### Project Docker files

| File | Role |
|------|------|
| `Dockerfile` | Recipe: Python 3.11, install requirements, run FastAPI on port 8000 |
| `docker-compose.yml` | Orchestrates 3 services: **api**, **dashboard**, **mlflow** |

### docker-compose services

```
api        → port 8000  (FastAPI forecasting)
dashboard  → port 8501  (Streamlit)
mlflow     → port 5000  (experiment tracking UI)
```

### Commands

```powershell
docker compose build    # Build images
docker compose up -d      # Start in background
docker compose down       # Stop
```

Volumes mount `./data`, `./models`, `./mlruns` so data persists outside containers.

---

## 10. AWS & Deployment Explained

| AWS service | Role in project |
|-------------|-----------------|
| **S3** | Store trained models and datasets (`S3_BUCKET` in `.env`) |
| **EC2** | GPU instance to train TFT and host API (`deployment/aws/deploy_ec2.sh`) |
| **Lambda** | Serverless trigger when new data arrives → call `/retrain` API |
| **CloudWatch** | Alarms when WMAPE too high or API errors (`cloudwatch_alarms.json`) |

**Flow:** New sales land in S3 → Lambda fires → API starts retraining → new model saved to S3 → EC2 loads new model.

---

## 11. MLOps & Monitoring

### MLflow (`mlruns/` folder)

- Every training run logs: metrics (WMAPE, MAE), parameters, model artifacts  
- Compare runs in UI: http://localhost:5000  

### Evidently (`src/monitoring/drift.py`)

- Compares **reference** (training) data vs **current** (production) data  
- Outputs HTML drift report in `reports/evidently/`  

### Auto-retraining (`src/monitoring/retraining.py`)

Retrain when:

1. WMAPE > threshold (default 25%)  
2. Drift detected  
3. New data flag set  

### Model registry (`src/monitoring/mlflow_registry.py`)

- Register model versions  
- Promote best model to **Production** stage  

---

## 12. How Data Flows Through the System

```
1. download_data.py
      ↓
   data/raw/*.parquet

2. ETLPipeline (clean, outliers, align dates, scale)
      ↓
   data/processed/ (optional save)

3. FeatureEngineeringPipeline (lags, rolling, Fourier...)
      ↓
   Feature matrix

4. train.py → ModelTrainer
      ↓
   models/*.pkl, models/tft/*.ckpt
   mlruns/ (experiments)
   reports/model_comparison.csv

5. inference.py OR FastAPI /forecast
      ↓
   JSON predictions

6. dashboards/app.py
      ↓
   Charts for humans
```

---

## 13. Every File in the Project

### Root files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `PROJECT_COMPLETE_GUIDE.md` | **This file** — full A–Z explanation |
| `how to run.txt` | Step-by-step run commands |
| `requirements.txt` | All Python package dependencies |
| `Dockerfile` | Container image for API |
| `docker-compose.yml` | Multi-container setup |
| `Makefile` | Shortcuts: `make train`, `make api`, etc. |
| `.env.example` | Template for secrets (copy to `.env`) |
| `.gitignore` | Files Git should not track (venv, data, secrets) |
| `pytest.ini` | Test configuration |

### `configs/`

| File | Purpose |
|------|---------|
| `config.yaml` | **Main config**: horizons, TFT params, paths, thresholds |
| `data_schema.yaml` | Data dictionary — column definitions for all tables |

### `data/`

| Path | Purpose |
|------|---------|
| `data/raw/` | Original / downloaded data |
| `data/raw/synthetic_sales.parquet` | Generated sales (your main dataset) |
| `data/raw/sku_metadata.parquet` | SKU attributes |
| `data/raw/dc_metadata.parquet` | Warehouse attributes |
| `data/processed/` | Cleaned data after ETL (optional) |
| `data/raw/.gitkeep` | Keeps empty folder in Git |

### `src/` — Core application code

#### `src/data/` — Data pipeline

| File | Purpose |
|------|---------|
| `download.py` | Download M5, Favorita, UCI; or generate synthetic |
| `synthetic.py` | Creates realistic fake FMCG sales data |
| `etl.py` | Clean, fill missing, outliers, time alignment, train/val/test split |
| `validation.py` | Checks required columns, nulls, duplicates, history length |
| `__init__.py` | Package exports |

#### `src/features/`

| File | Purpose |
|------|---------|
| `engineering.py` | Lags, rolling mean/std, EMA, Fourier, holidays, promo, inventory features |

#### `src/models/`

| File | Purpose |
|------|---------|
| `base.py` | Abstract class all models implement (fit, predict, save) |
| `baseline.py` | Moving Average, ARIMA, SARIMA, ETS |
| `ml_models.py` | XGBoost, LightGBM, CatBoost, Random Forest |
| `deep_learning.py` | LSTM, GRU, Seq2Seq, Attention (PyTorch Lightning) |
| `tft_model.py` | **Temporal Fusion Transformer** — main production model |

#### `src/training/`

| File | Purpose |
|------|---------|
| `trainer.py` | Orchestrates full pipeline; logs to MLflow; saves comparison |
| `cross_validation.py` | Time-series expanding-window CV |

#### `src/inference/`

| File | Purpose |
|------|---------|
| `predictor.py` | Loads model; preprocesses input; returns predictions |

#### `src/explainability/`

| File | Purpose |
|------|---------|
| `shap_explainer.py` | SHAP global/local explanations for ML models |
| `attention_viz.py` | TFT attention heatmap plots |

#### `src/monitoring/`

| File | Purpose |
|------|---------|
| `drift.py` | Evidently / statistical drift detection |
| `retraining.py` | Decides when to retrain; runs training with retries |
| `mlflow_registry.py` | Register and promote models in MLflow |

#### `src/api/`

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app: `/forecast`, `/health`, `/retrain`, etc. |
| `schemas.py` | Pydantic models for request/response validation |
| `auth.py` | Bearer token authentication |

#### `src/utils/`

| File | Purpose |
|------|---------|
| `config.py` | Load YAML + `.env` settings |
| `metrics.py` | MAE, RMSE, MAPE, WMAPE, SMAPE calculations |
| `logger.py` | Logging setup |
| `seed.py` | Reproducible random seeds |

### `scripts/` — Runnable entry points

| File | Purpose |
|------|---------|
| `download_data.py` | CLI: generate or download datasets |
| `train.py` | CLI: train models by phase |
| `inference.py` | CLI: batch predictions from file |
| `generate_report.py` | CLI: create model comparison markdown report |

### `dashboards/`

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (5 tabs) |

### `deployment/aws/`

| File | Purpose |
|------|---------|
| `deploy_ec2.sh` | Script to deploy API on EC2, sync models to S3 |
| `lambda_retrain.py` | AWS Lambda handler to trigger retraining |
| `cloudwatch_alarms.json` | Example alarm definitions |

### `docs/`

| File | Purpose |
|------|---------|
| `architecture.md` | System architecture with Mermaid diagrams |
| `data_flow.md` | Sequence diagram of data through pipeline |

### `notebooks/`

| File | Purpose |
|------|---------|
| `01_data_exploration.md` | Guide for exploring data in Jupyter |
| `02_preprocessing.md` | Guide for ETL steps |

### `tests/`

| File | Purpose |
|------|---------|
| `test_metrics.py` | Tests for MAE, WMAPE, etc. |
| `test_etl.py` | Tests for validation and ETL |
| `test_features.py` | Tests for feature engineering |
| `test_api.py` | Tests for FastAPI health and forecast |

### `.github/workflows/`

| File | Purpose |
|------|---------|
| `ci.yml` | GitHub Actions: lint, test, build Docker on push |

### Generated at runtime (not in Git)

| Path | Purpose |
|------|---------|
| `.venv/` | Python virtual environment |
| `mlruns/` | MLflow experiment logs |
| `models/` | Saved model files (.pkl, .ckpt) |
| `reports/` | Comparison CSV, SHAP plots, drift HTML |
| `lightning_logs/` | TensorBoard logs for deep models |

---

## 14. What You Can Say in Interviews / Resume

**Project title:** AI-Powered Multi-Horizon Demand Forecasting for FMCG Supply Chains

**Bullet points:**

- Built end-to-end demand forecasting platform for 500+ SKUs across 50 distribution centers  
- Implemented statistical baselines, gradient boosting, LSTM/GRU, and **Temporal Fusion Transformer**  
- Achieved quantile multi-horizon forecasts with static and time-varying features  
- Deployed **FastAPI** inference service with auth, batch prediction, and Swagger docs  
- Integrated **MLflow**, **Evidently** drift detection, and automated retraining pipeline  
- Containerized with **Docker**; designed **AWS** deployment (S3, EC2, Lambda, CloudWatch)  
- Built **Streamlit** dashboard for planners with SHAP explainability  

**Skills demonstrated:** Python, PyTorch, Time Series, MLOps, FastAPI, Docker, AWS, SHAP, CI/CD

---

## 15. Quick Commands Reference

```powershell
# Setup
cd c:\Users\basav\OneDrive\Desktop\fmcg-demand-forecasting
.\.venv\Scripts\activate

# Data
python scripts/download_data.py --dataset synthetic

# Train
python scripts/train.py --phases baseline ml tft

# API
uvicorn src.api.main:app --reload --port 8000

# Dashboard
streamlit run dashboards/app.py

# Tests
pytest tests/ -v

# Docker
docker compose up -d
```

---

## Summary

| Question | Answer |
|----------|--------|
| **Why build it?** | Learn and demonstrate enterprise supply-chain AI like top retailers use |
| **What's the use?** | Predict SKU demand per warehouse to optimize inventory |
| **Best model?** | Temporal Fusion Transformer (TFT) for multi-horizon + rich features |
| **How to interact?** | API for systems, Streamlit for humans |
| **Why Docker?** | Reproducible deployment anywhere |
| **Default data?** | Synthetic FMCG sales in `data/raw/synthetic_sales.parquet` |
| **Is it complete?** | Yes — all 10 phases implemented with tests and docs |

---

*Last updated: May 2026 — FMCG Demand Forecasting Project*
