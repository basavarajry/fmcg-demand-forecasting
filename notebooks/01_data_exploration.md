# Notebook 01: Data Exploration

Run in Jupyter after installing requirements:

```python
import sys
sys.path.insert(0, "..")

from src.data.download import DatasetDownloader
from src.utils.config import PROJECT_ROOT, load_config

config = load_config()
downloader = DatasetDownloader(PROJECT_ROOT / "data/raw")
df = downloader.load_sales("synthetic")
df.head()
df.groupby(["sku_id", "dc_id"]).size().describe()
```

See `configs/data_schema.yaml` for the full data dictionary.
