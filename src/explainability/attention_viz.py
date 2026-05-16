"""TFT attention visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttentionVisualizer:
    """Visualize TFT attention weights."""

    @staticmethod
    def plot_attention_heatmap(
        attention_weights: np.ndarray,
        output_path: Path,
        title: str = "TFT Attention Weights",
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(attention_weights, aspect="auto", cmap="YlOrRd")
        ax.set_title(title)
        ax.set_xlabel("Encoder Time Step")
        ax.set_ylabel("Decoder Time Step")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info("Attention heatmap saved to %s", output_path)

    @staticmethod
    def feature_contribution_bar(
        contributions: Dict[str, float],
        output_path: Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        names = list(contributions.keys())
        values = list(contributions.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(names, values, color="steelblue")
        ax.set_xlabel("Contribution")
        ax.set_title("Forecast Driver Analysis")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
