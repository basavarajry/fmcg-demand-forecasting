"""Unit tests for ETL pipeline."""

import pandas as pd
import pytest

from src.data.synthetic import SyntheticDataGenerator
from src.data.validation import DataValidator
from src.data.etl import ETLPipeline
from src.utils.config import load_config


@pytest.fixture
def sample_df():
    gen = SyntheticDataGenerator(n_skus=2, n_dcs=2, n_days=120)
    return gen.generate()["sales_history"]


def test_validation_passes(sample_df):
    validator = DataValidator(min_history_days=30)
    result = validator.validate(sample_df)
    assert result.passed


def test_etl_pipeline(sample_df):
    config = load_config()
    etl = ETLPipeline(config)
    processed = etl.run(sample_df)
    assert "demand" in processed.columns
    assert processed["demand"].isnull().sum() == 0


def test_train_val_test_split(sample_df):
    config = load_config()
    etl = ETLPipeline(config)
    processed = etl.run(sample_df)
    train, val, test = etl.train_val_test_split(processed)
    assert len(train) > len(test)
    assert train["date"].max() < test["date"].min()
