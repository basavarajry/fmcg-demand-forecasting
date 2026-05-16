# System Architecture

## High-Level Architecture

```mermaid
flowchart TB
    subgraph DataSources["Data Sources"]
        Sales[Sales History]
        Promo[Promotions]
        Holiday[Holidays]
        Inv[Inventory]
        Comp[Competitor Pricing]
        Web[Web Trends]
        Macro[Macro Indicators]
    end

    subgraph Pipeline["Data Pipeline"]
        ETL[ETL + Validation]
        FE[Feature Engineering]
    end

    subgraph Models["Model Layer"]
        BL[Statistical Baselines]
        ML[XGBoost / LightGBM / CatBoost]
        DL[LSTM / GRU / Seq2Seq]
        TFT[Temporal Fusion Transformer]
    end

    subgraph MLOps["MLOps"]
        MLflow[MLflow Tracking]
        Evidently[Evidently Drift]
        Retrain[Auto Retraining]
    end

    subgraph Serving["Serving"]
        API[FastAPI]
        Dash[Streamlit Dashboard]
        S3[(AWS S3 Models)]
    end

    DataSources --> ETL --> FE
    FE --> BL & ML & DL & TFT
    BL & ML & DL & TFT --> MLflow
    TFT --> API
    ML --> API
    Evidently --> Retrain
    Retrain --> TFT
    API --> Dash
    TFT --> S3
```

## Component Responsibilities

| Component | Technology | Role |
|-----------|------------|------|
| ETL Pipeline | Pandas, scikit-learn | Clean, align, scale time series |
| Feature Store | Custom pipeline | Lags, rolling, Fourier, exogenous |
| TFT Trainer | pytorch-forecasting, Lightning | Multi-horizon quantile forecasts |
| SHAP | shap | ML model explainability |
| API | FastAPI, Pydantic | Production inference |
| Monitoring | Evidently, CloudWatch | Drift + accuracy alerts |
| Registry | MLflow | Experiment + model versioning |

## Deployment Topology (AWS)

```mermaid
flowchart LR
    S3Data[(S3 Data)] --> Lambda[Lambda Trigger]
    Lambda --> EC2[EC2 GPU Instance]
    EC2 --> API[FastAPI Container]
    API --> CW[CloudWatch Metrics]
    CW --> Lambda
    EC2 --> S3Models[(S3 Models)]
```
