# Data Flow

```mermaid
sequenceDiagram
    participant Raw as Raw Data (S3/Local)
    participant DL as DatasetDownloader
    participant ETL as ETLPipeline
    participant VAL as DataValidator
    participant FE as FeatureEngineering
    participant TR as ModelTrainer
    participant MLF as MLflow
    participant API as FastAPI

    Raw->>DL: Download M5/Favorita/Synthetic
    DL->>VAL: Validate schema
    VAL->>ETL: Pass validation
    ETL->>FE: Processed parquet
    FE->>TR: Feature matrix
    TR->>MLF: Log metrics & artifacts
    TR->>API: Deploy best model
    API-->>Client: Forecast JSON
```

## Feature Categories (TFT)

| Type | Examples |
|------|----------|
| Static categorical | sku_id, dc_id, category, brand, region |
| Time-varying known | promo_flag, is_holiday, day_of_week |
| Time-varying unknown | demand, inventory_ratio, web_trend |

## Retraining Triggers

1. WMAPE > threshold (default 25%)
2. Evidently data drift detected
3. New data landed in S3 (Lambda event)
