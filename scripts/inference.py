#!/usr/bin/env python
"""Batch inference script."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predictor import ForecastPredictor
from src.utils.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet/csv path")
    parser.add_argument("--output", default="reports/predictions.json")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    predictor = ForecastPredictor(model_name=args.model)
    result = predictor.predict(df, horizon=args.horizon)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Predictions saved to {out}")


if __name__ == "__main__":
    main()
