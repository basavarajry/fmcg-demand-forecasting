"""Synthetic FMCG demand dataset generator for development and testing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """Generate realistic multi-SKU, multi-DC demand time series."""

    def __init__(
        self,
        n_skus: int = 100,
        n_dcs: int = 20,
        n_days: int = 730,
        seed: int = 42,
    ) -> None:
        self.n_skus = n_skus
        self.n_dcs = n_dcs
        self.n_days = n_days
        self.rng = np.random.default_rng(seed)

    def generate(self, start_date: str = "2022-01-01") -> dict[str, pd.DataFrame]:
        dates = pd.date_range(start_date, periods=self.n_days, freq="D")
        skus = [f"SKU_{i:04d}" for i in range(self.n_skus)]
        dcs = [f"DC_{i:02d}" for i in range(self.n_dcs)]
        categories = ["Beverages", "Snacks", "Dairy", "Personal Care", "Household"]
        brands = ["BrandA", "BrandB", "BrandC", "BrandD"]
        regions = ["North", "South", "East", "West", "Central"]

        sku_meta = pd.DataFrame(
            {
                "sku_id": skus,
                "category": self.rng.choice(categories, self.n_skus),
                "brand": self.rng.choice(brands, self.n_skus),
                "subcategory": [f"sub_{i % 10}" for i in range(self.n_skus)],
                "is_perishable": self.rng.choice([0, 1], self.n_skus, p=[0.7, 0.3]),
            }
        )

        dc_meta = pd.DataFrame(
            {
                "dc_id": dcs,
                "region": self.rng.choice(regions, self.n_dcs),
                "capacity": self.rng.integers(5000, 50000, self.n_dcs),
            }
        )

        records = []
        for sku in skus:
            base = self.rng.uniform(5, 200)
            trend = self.rng.uniform(-0.02, 0.05)
            for dc in dcs:
                dc_factor = self.rng.uniform(0.5, 1.5)
                for i, d in enumerate(dates):
                    seasonal = 1 + 0.3 * np.sin(2 * np.pi * i / 365)
                    weekly = 1 + 0.15 * np.sin(2 * np.pi * i / 7)
                    noise = self.rng.normal(0, base * 0.1)
                    promo = 1.0
                    if self.rng.random() < 0.05:
                        promo = self.rng.uniform(1.2, 1.8)
                    demand = max(0, base * dc_factor * seasonal * weekly * promo * (1 + trend * i / 365) + noise)
                    records.append(
                        {
                            "date": d,
                            "sku_id": sku,
                            "dc_id": dc,
                            "demand": round(demand, 2),
                            "promo_flag": int(promo > 1.0),
                            "discount_pct": round((promo - 1) * 50, 1) if promo > 1 else 0,
                            "inventory_ratio": self.rng.uniform(0.3, 2.0),
                            "competitor_price_idx": self.rng.uniform(0.85, 1.15),
                            "web_trend": self.rng.uniform(20, 100),
                            "is_holiday": int(d.dayofweek >= 5 and self.rng.random() < 0.3),
                        }
                    )

        sales = pd.DataFrame(records)
        sales = sales.merge(sku_meta, on="sku_id").merge(dc_meta, on="dc_id")
        logger.info("Generated synthetic data: %d rows", len(sales))
        return {
            "sales_history": sales,
            "sku_metadata": sku_meta,
            "dc_metadata": dc_meta,
        }

    def save(self, output_dir: Path, **kwargs) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = self.generate(**kwargs)
        path = output_dir / "synthetic_sales.parquet"
        data["sales_history"].to_parquet(path, index=False)
        data["sku_metadata"].to_parquet(output_dir / "sku_metadata.parquet", index=False)
        data["dc_metadata"].to_parquet(output_dir / "dc_metadata.parquet", index=False)
        return path
