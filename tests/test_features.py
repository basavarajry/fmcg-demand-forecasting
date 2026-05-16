"""Tests for feature engineering."""

from src.data.synthetic import SyntheticDataGenerator
from src.features.engineering import FeatureEngineeringPipeline
from src.utils.config import load_config


def test_feature_engineering():
    gen = SyntheticDataGenerator(n_skus=2, n_dcs=1, n_days=60)
    df = gen.generate()["sales_history"]
    pipeline = FeatureEngineeringPipeline(load_config())
    featured = pipeline.transform(df)
    assert "lag_7" in featured.columns
    assert "roll_mean_7" in featured.columns
    assert "fourier_sin_3" in featured.columns
