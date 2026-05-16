#!/usr/bin/env python
"""Main training script."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import ModelTrainer
from src.utils.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Train demand forecasting models")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["baseline", "ml", "tft"],
        choices=["baseline", "ml", "deep", "tft"],
    )
    args = parser.parse_args()

    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)

    trainer = ModelTrainer()
    comparison = trainer.run_full_pipeline(phases=args.phases)
    print("\n=== Model Comparison ===")
    print(comparison.to_string())


if __name__ == "__main__":
    main()
