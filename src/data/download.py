"""Automatic dataset download for M5, Favorita, and UCI time series."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

DatasetType = Literal["m5", "favorita", "uci", "synthetic"]


class DatasetDownloader:
    """Download and extract public forecasting datasets."""

    M5_URL = "https://zenodo.org/record/4679600/files/m5-forecasting-accuracy.zip"
    FAVORITA_KAGGLE = "competitions/download/13535"

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download(self, dataset: DatasetType, force: bool = False) -> Path:
        handlers = {
            "synthetic": self._synthetic,
            "m5": self._download_m5,
            "favorita": self._download_favorita,
            "uci": self._download_uci,
        }
        if dataset not in handlers:
            raise ValueError(f"Unknown dataset: {dataset}")
        return handlers[dataset](force=force)

    def _synthetic(self, force: bool = False) -> Path:
        from src.data.synthetic import SyntheticDataGenerator

        path = self.raw_dir / "synthetic_sales.parquet"
        if path.exists() and not force:
            return path
        gen = SyntheticDataGenerator(n_skus=50, n_dcs=10, n_days=365)
        return gen.save(self.raw_dir)

    def _download_m5(self, force: bool = False) -> Path:
        """Download M5 from Zenodo (large ~5GB). Samples subset after extract."""
        dest = self.raw_dir / "m5"
        dest.mkdir(exist_ok=True)
        marker = dest / ".downloaded"
        if marker.exists() and not force:
            return dest

        zip_path = self.raw_dir / "m5-forecasting-accuracy.zip"
        if not zip_path.exists():
            logger.info("Downloading M5 dataset (this may take a while)...")
            try:
                import urllib.request

                urllib.request.urlretrieve(self.M5_URL, zip_path)
            except Exception as e:
                logger.warning("M5 download failed: %s. Using synthetic fallback.", e)
                return self._synthetic(force=True)

        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            marker.touch()
        return dest

    def _download_favorita(self, force: bool = False) -> Path:
        dest = self.raw_dir / "favorita"
        dest.mkdir(exist_ok=True)
        marker = dest / ".downloaded"
        if marker.exists() and not force:
            return dest

        try:
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", "favorita-grocery-sales-forecasting", "-p", str(dest)],
                check=True,
                capture_output=True,
            )
            for z in dest.glob("*.zip"):
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(dest)
            marker.touch()
        except Exception as e:
            logger.warning("Favorita download failed (Kaggle API required): %s", e)
            return self._synthetic(force=True)
        return dest

    def _download_uci(self, force: bool = False) -> Path:
        """Load UCI Online Retail as proxy time series."""
        dest = self.raw_dir / "uci"
        dest.mkdir(exist_ok=True)
        out = dest / "uci_retail.parquet"
        if out.exists() and not force:
            return out

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        try:
            df = pd.read_excel(url)
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            daily = (
                df.groupby([df["InvoiceDate"].dt.date, "StockCode"])["Quantity"]
                .sum()
                .reset_index()
            )
            daily.columns = ["date", "sku_id", "demand"]
            daily["dc_id"] = "DC_00"
            daily.to_parquet(out, index=False)
        except Exception as e:
            logger.warning("UCI download failed: %s", e)
            return self._synthetic(force=True)
        return out

    def load_sales(self, dataset: DatasetType, max_rows: Optional[int] = None) -> pd.DataFrame:
        path = self.download(dataset)
        if dataset == "synthetic" or path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif dataset == "m5":
            df = self._load_m5_sample(path)
        elif dataset == "favorita":
            df = self._load_favorita_sample(path)
        else:
            df = pd.read_parquet(path)

        if max_rows and len(df) > max_rows:
            df = df.sample(max_rows, random_state=42)
        return df

    def _load_m5_sample(self, m5_dir: Path, n_series: int = 500) -> pd.DataFrame:
        sales_files = list(m5_dir.rglob("sales_train_evaluation.csv"))
        if not sales_files:
            return pd.read_parquet(self.raw_dir / "synthetic_sales.parquet")
        df = pd.read_csv(sales_files[0])
        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        value_cols = [c for c in df.columns if c.startswith("d_")]
        melted = df[id_cols + value_cols].melt(
            id_vars=id_cols, var_name="day", value_name="demand"
        )
        melted["date"] = pd.to_datetime(melted["day"].str.replace("d_", ""), format="%j", errors="coerce")
        melted = melted.rename(columns={"item_id": "sku_id", "store_id": "dc_id"})
        series = melted.groupby(["sku_id", "dc_id"]).ngroup()
        keep = melted.groupby(["sku_id", "dc_id"]).size().nlargest(n_series).index
        melted = melted.set_index(["sku_id", "dc_id"]).loc[keep].reset_index()
        return melted[["date", "sku_id", "dc_id", "demand", "cat_id", "dept_id", "state_id"]].dropna()

    def _load_favorita_sample(self, favorita_dir: Path) -> pd.DataFrame:
        train_files = list(favorita_dir.rglob("train.csv"))
        if not train_files:
            return pd.read_parquet(self.raw_dir / "synthetic_sales.parquet")
        df = pd.read_csv(train_files[0], nrows=500_000)
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"item_nbr": "sku_id", "store_nbr": "dc_id", "unit_sales": "demand"})
        df["dc_id"] = df["dc_id"].astype(str)
        df["sku_id"] = df["sku_id"].astype(str)
        return df
