# Notebook 02: Data Preprocessing

```python
from src.data.etl import ETLPipeline
from src.data.validation import DataValidator
from src.utils.config import load_config

config = load_config()
# ... load df ...
validator = DataValidator()
validator.validate(df)

etl = ETLPipeline(config)
processed = etl.run(df)
train, val, test = etl.train_val_test_split(processed)
etl.save_processed(processed, "data/processed/sales.parquet")
```
