"""
Visualization: training loss and metric plots.

Generates:
  - Train/val loss curves
  - Train/val BLEU-100 curves
  - Train/val CHRF++-100 curves
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    history: dict[str, list[float]],
    output_dir: str,
) -> list[str]:
    """Plot and save training curves.

    Expected keys in history:
      - train_loss, val_loss
      - train_bleu, val_bleu
      - train_chrf, val_chrf

    Args:
        history: Dict of metric_name → list of per-epoch values.
        output_dir: Directory to save plots.

    Returns:
        List of saved plot file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # Define which plots to make
    plot_specs = [
        {
            "title": "Training & Validation Loss",
            "train_key": "train_loss",
            "val_key": "val_loss",
            "ylabel": "Loss",
            "filename": "loss_curves.png",
        },
        {
            "title": "BLEU-100 Score",
            "train_key": "train_bleu",
            "val_key": "val_bleu",
            "ylabel": "BLEU-100",
            "filename": "bleu_curves.png",
        },
        {
            "title": "CHRF++-100 Score",
            "train_key": "train_chrf",
            "val_key": "val_chrf",
            "ylabel": "CHRF++-100",
            "filename": "chrf_curves.png",
        },
    ]

    for spec in plot_specs:
        train_vals = history.get(spec["train_key"], [])
        val_vals = history.get(spec["val_key"], [])

        if not train_vals and not val_vals:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, max(len(train_vals), len(val_vals)) + 1)

        if train_vals:
            ax.plot(epochs[: len(train_vals)], train_vals, "b-o", label="Train", markersize=4)
        if val_vals:
            ax.plot(epochs[: len(val_vals)], val_vals, "r-o", label="Validation", markersize=4)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(spec["ylabel"], fontsize=12)
        ax.set_title(spec["title"], fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, spec["filename"])
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)
        print(f"  Plot saved: {path}")

    return saved_paths


def plot_attention(
    attention_weights: list[list[float]],
    src_tokens: list[str],
    tgt_tokens: list[str],
    output_path: str,
) -> None:
    """Plot an attention heatmap for a single sentence pair.

    Args:
        attention_weights: (tgt_len, src_len) attention matrix.
        src_tokens: Source token strings.
        tgt_tokens: Target token strings.
        output_path: Path to save the plot.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 8))

    attn = np.array(attention_weights)
    im = ax.imshow(attn, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=8)

    ax.set_xlabel("Source", fontsize=12)
    ax.set_ylabel("Target", fontsize=12)
    ax.set_title("Attention Weights", fontsize=14)

    fig.colorbar(im, ax=ax)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
