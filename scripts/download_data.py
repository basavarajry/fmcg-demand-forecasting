#!/usr/bin/env python
"""Download or generate datasets."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import DatasetDownloader
from src.utils.config import load_config, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Download forecasting datasets")
    parser.add_argument("--dataset", choices=["synthetic", "m5", "favorita", "uci"], default="synthetic")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    downloader = DatasetDownloader(raw_dir)
    path = downloader.download(args.dataset, force=args.force)
    print(f"Dataset ready at: {path}")


if __name__ == "__main__":
    main()
