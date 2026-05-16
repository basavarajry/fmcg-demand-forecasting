"""Time-series cross-validation."""

from __future__ import annotations

from typing import Generator, List, Tuple

import pandas as pd


class TimeSeriesCV:
    """Expanding window time-series cross-validation."""

    def __init__(self, n_splits: int = 5, test_size: int = 28, gap: int = 0) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, df: pd.DataFrame, date_col: str = "date") -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        dates = sorted(df[date_col].unique())
        n = len(dates)
        min_train = n - self.n_splits * self.test_size - self.gap

        for i in range(self.n_splits):
            test_end_idx = n - i * self.test_size
            test_start_idx = test_end_idx - self.test_size
            train_end_idx = test_start_idx - self.gap
            if train_end_idx < min_train // self.n_splits:
                break
            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            train = df[df[date_col].isin(train_dates)]
            test = df[df[date_col].isin(test_dates)]
            yield train, test
