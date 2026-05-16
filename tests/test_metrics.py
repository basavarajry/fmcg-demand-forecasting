"""Unit tests for forecast metrics."""

import numpy as np
import pytest

from src.utils.metrics import ForecastMetrics


def test_mae():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    assert ForecastMetrics.mae(y_true, y_pred) == pytest.approx(2.333, rel=0.01)


def test_wmape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    wmape = ForecastMetrics.wmape(y_true, y_pred)
    assert 0 < wmape < 20


def test_compute_all():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([11, 19, 31])
    metrics = ForecastMetrics.compute_all(y_true, y_pred)
    assert set(metrics.keys()) == {"mae", "rmse", "mape", "wmape", "smape"}


def test_improvement_pct():
    imp = ForecastMetrics.improvement_pct(20.0, 16.0)
    assert imp == pytest.approx(20.0)
