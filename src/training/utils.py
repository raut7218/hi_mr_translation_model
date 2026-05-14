"""
Training utilities: seed, optimizer, scheduler, parameter counting.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn

from src.config import TrainingConfig


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic mode (slight perf hit, better reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the device used by this project."""
    device = torch.device("cpu")
    print("Using CPU")
    return device


def get_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build a simple LR scheduler (ReduceLROnPlateau)."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  Loaded checkpoint: {path} (epoch {checkpoint['epoch']})")
    return checkpoint
