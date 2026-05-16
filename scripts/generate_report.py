#!/usr/bin/env python
"""Generate model comparison and SHAP reports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.config import PROJECT_ROOT


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    comparison_path = reports_dir / "model_comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path, index_col=0)
        md = ["# Model Comparison Report\n", df.to_markdown(), "\n"]
        if "tft" in df.index and "moving_average" in df.index:
            imp = (df.loc["moving_average", "wmape"] - df.loc["tft", "wmape"]) / df.loc["moving_average", "wmape"] * 100
            md.append(f"\n**TFT WMAPE improvement over Moving Average: {imp:.1f}%**\n")
        (reports_dir / "model_comparison_report.md").write_text("\n".join(md), encoding="utf-8")
        print(f"Report saved to {reports_dir / 'model_comparison_report.md'}")
    else:
        print("Run training first: python scripts/train.py")


if __name__ == "__main__":
    main()
