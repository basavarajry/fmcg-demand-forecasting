"""Advanced time-series feature engineering pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import holidays
except ImportError:
    holidays = None

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineeringPipeline:
    """Automated feature generation for demand forecasting."""

    def __init__(self, config: Dict[str, Any]) -> None:
        feat_cfg = config.get("features", {})
        data_cfg = config.get("data", {})
        self.lags = feat_cfg.get("lags", [1, 7, 14, 28])
        self.rolling_windows = feat_cfg.get("rolling_windows", [7, 14, 28])
        self.fourier_orders = feat_cfg.get("fourier_orders", [3, 7, 14])
        self.date_col = data_cfg.get("date_col", "date")
        self.target_col = data_cfg.get("target_col", "demand")
        self.group_cols = data_cfg.get("group_cols", ["sku_id", "dc_id"])
        self.include_holidays = feat_cfg.get("include_holidays", True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = self._date_features(df)
        df = self._lag_features(df)
        df = self._rolling_features(df)
        df = self._ema_features(df)
        df = self._fourier_features(df)
        df = self._volatility_features(df)
        if self.include_holidays:
            df = self._holiday_features(df)
        df = self._promotion_features(df)
        df = self._inventory_features(df)
        df = self._external_features(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        logger.info("Feature engineering complete: %d columns", len(df.columns))
        return df

    def _date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df[self.date_col]
        df["day_of_week"] = d.dt.dayofweek
        df["day_of_month"] = d.dt.day
        df["week_of_year"] = d.dt.isocalendar().week.astype(int)
        df["month"] = d.dt.month
        df["quarter"] = d.dt.quarter
        df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
        df["is_month_start"] = d.dt.is_month_start.astype(int)
        df["is_month_end"] = d.dt.is_month_end.astype(int)
        return df

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in self.lags:
            df[f"lag_{lag}"] = df.groupby(self.group_cols)[self.target_col].shift(lag)
        return df

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in self.rolling_windows:
            g = df.groupby(self.group_cols)[self.target_col]
            df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            df[f"roll_std_{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).std())
        return df

    def _ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for span in [7, 14, 28]:
            df[f"ema_{span}"] = df.groupby(self.group_cols)[self.target_col].transform(
                lambda s: s.shift(1).ewm(span=span, adjust=False).mean()
            )
        return df

    def _fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        t = (df[self.date_col] - df[self.date_col].min()).dt.days.values.astype(float)
        for order in self.fourier_orders:
            df[f"fourier_sin_{order}"] = np.sin(2 * np.pi * order * t / 365.25)
            df[f"fourier_cos_{order}"] = np.cos(2 * np.pi * order * t / 365.25)
        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["demand_volatility_7"] = df.groupby(self.group_cols)[self.target_col].transform(
            lambda s: s.shift(1).rolling(7, min_periods=3).std()
        )
        df["demand_cv_28"] = df.groupby(self.group_cols)[self.target_col].transform(
            lambda s: s.shift(1).rolling(28, min_periods=7).std()
            / (s.shift(1).rolling(28, min_periods=7).mean() + 1e-8)
        )
        return df

    def _holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "is_holiday" in df.columns:
            return df
        if holidays is not None:
            us_holidays = holidays.US(years=range(2020, 2027))
            df["is_holiday"] = df[self.date_col].dt.date.apply(lambda x: x in us_holidays).astype(int)
        else:
            df["is_holiday"] = 0
        return df

    def _promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "promo_flag" not in df.columns:
            df["promo_flag"] = 0
        if "discount_pct" not in df.columns:
            df["discount_pct"] = 0
        df["promo_intensity"] = df["promo_flag"] * (1 + df["discount_pct"] / 100)
        return df

    def _inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "inventory_ratio" not in df.columns:
            df["inventory_ratio"] = 1.0
        df["stock_pressure"] = 1 / (df["inventory_ratio"] + 0.1)
        return df

    def _external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, default in [("competitor_price_idx", 1.0), ("web_trend", 50.0)]:
            if col not in df.columns:
                df[col] = default
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = {self.date_col, self.target_col} | set(self.group_cols)
        return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
