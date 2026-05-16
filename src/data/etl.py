"""ETL pipeline: cleaning, outlier handling, alignment, scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.data.validation import DataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ETLPipeline:
    """End-to-end data preprocessing pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        data_cfg = config.get("data", {})
        prep_cfg = config.get("preprocessing", {})
        self.date_col = data_cfg.get("date_col", "date")
        self.target_col = data_cfg.get("target_col", "demand")
        self.group_cols = data_cfg.get("group_cols", ["sku_id", "dc_id"])
        self.fill_method = prep_cfg.get("fill_method", "forward")
        self.outlier_method = prep_cfg.get("outlier_method", "iqr")
        self.outlier_iqr_factor = prep_cfg.get("outlier_iqr_factor", 1.5)
        self.scale_method = prep_cfg.get("scale_method", "robust")
        self.validator = DataValidator(
            date_col=self.date_col,
            target_col=self.target_col,
            group_cols=self.group_cols,
            min_history_days=data_cfg.get("min_history_days", 90),
        )
        self._scalers: Dict[str, Any] = {}

    def run(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        validation = self.validator.validate(df)
        if not validation.passed:
            raise ValueError(f"Data validation failed: {validation.errors}")

        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.group_cols + [self.date_col])
        df = self._handle_missing(df)
        df = self._detect_outliers(df)
        df = self._align_time(df)
        if fit:
            df = self._fit_scale(df)
        else:
            df = self._transform_scale(df)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == self.target_col:
                df[col] = df.groupby(self.group_cols)[col].transform(
                    lambda s: s.ffill().bfill().fillna(0)
                )
            elif self.fill_method == "forward":
                df[col] = df.groupby(self.group_cols)[col].ffill().bfill()
            else:
                df[col] = df.groupby(self.group_cols)[col].transform(lambda s: s.fillna(s.median()))
        return df

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_method != "iqr":
            return df

        def clip_iqr(s: pd.Series) -> pd.Series:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low = q1 - self.outlier_iqr_factor * iqr
            high = q3 + self.outlier_iqr_factor * iqr
            return s.clip(lower=max(0, low), upper=high)

        df[self.target_col] = df.groupby(self.group_cols)[self.target_col].transform(clip_iqr)
        return df

    def _align_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex each series to continuous daily frequency."""
        frames = []
        for keys, grp in df.groupby(self.group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            idx = pd.date_range(grp[self.date_col].min(), grp[self.date_col].max(), freq="D")
            grp = grp.set_index(self.date_col).reindex(idx)
            grp.index.name = self.date_col
            grp = grp.reset_index()
            for i, k in enumerate(self.group_cols):
                grp[k] = keys[i] if len(self.group_cols) > 1 else keys
            grp[self.target_col] = grp[self.target_col].fillna(0)
            frames.append(grp)
        return pd.concat(frames, ignore_index=True)

    def _get_scaler(self):
        if self.scale_method == "standard":
            return StandardScaler()
        return RobustScaler()

    def _fit_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        scale_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != self.target_col]
        for col in scale_cols[:10]:  # limit for memory
            scaler = self._get_scaler()
            df[col] = scaler.fit_transform(df[[col]])
            self._scalers[col] = scaler
        return df

    def _transform_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, scaler in self._scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])
        return df

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        max_date = df[self.date_col].max()
        n_days = (df[self.date_col].max() - df[self.date_col].min()).days + 1
        test_days = int(n_days * (1 - train_ratio - val_ratio))
        val_days = int(n_days * val_ratio)
        test_start = max_date - pd.Timedelta(days=test_days - 1)
        val_start = test_start - pd.Timedelta(days=val_days)

        train = df[df[self.date_col] < val_start]
        val = df[(df[self.date_col] >= val_start) & (df[self.date_col] < test_start)]
        test = df[df[self.date_col] >= test_start]
        logger.info("Split: train=%d, val=%d, test=%d rows", len(train), len(val), len(test))
        return train, val, test

    def save_processed(self, df: pd.DataFrame, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Saved processed data to %s", path)
