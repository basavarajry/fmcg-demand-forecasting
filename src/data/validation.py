"""Data validation and quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DataValidator:
    """Validate time series data before training."""

    REQUIRED_COLS = ["date", "sku_id", "dc_id", "demand"]

    def __init__(
        self,
        date_col: str = "date",
        target_col: str = "demand",
        group_cols: Optional[List[str]] = None,
        min_history_days: int = 90,
    ) -> None:
        self.date_col = date_col
        self.target_col = target_col
        self.group_cols = group_cols or ["sku_id", "dc_id"]
        self.min_history_days = min_history_days

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult(passed=True)

        for col in self.REQUIRED_COLS:
            if col not in df.columns:
                result.errors.append(f"Missing required column: {col}")
                result.passed = False

        if not result.passed:
            return result

        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        null_pct = df[self.target_col].isnull().mean()
        if null_pct > 0.1:
            result.warnings.append(f"Target has {null_pct:.1%} null values")

        neg = (df[self.target_col] < 0).sum()
        if neg > 0:
            result.warnings.append(f"{neg} negative demand values found")

        dup = df.duplicated(subset=self.group_cols + [self.date_col]).sum()
        if dup > 0:
            result.warnings.append(f"{dup} duplicate date-series rows")

        history = df.groupby(self.group_cols)[self.date_col].agg(["min", "max", "count"])
        short = (history["count"] < self.min_history_days).sum()
        if short > 0:
            result.warnings.append(f"{short} series shorter than {self.min_history_days} days")

        if result.errors:
            result.passed = False
        logger.info("Validation: passed=%s, errors=%d, warnings=%d", result.passed, len(result.errors), len(result.warnings))
        return result
