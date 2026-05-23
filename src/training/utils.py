"""
Training utilities: seed, optimizer, scheduler, parameter counting.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from src.config import TrainingConfig


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def init_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    """Initialize torch.distributed for DDP."""
    if dist.is_initialized():
        return
    if world_size <= 1:
        return
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed() -> None:
    """Tear down the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def reduce_tensor_mean(value: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor and return the mean across ranks."""
    if not is_distributed():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= get_world_size()
    return value


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the device used by this project."""
    preference = (device_preference or "auto").lower()
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preference in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference: {device_preference}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        name = torch.cuda.get_device_name(device)
        print(f"Using CUDA device: {name}")
    else:
        cpu_count = os.cpu_count() or 1
        target_threads = max(1, int(cpu_count * 0.8))
        torch.set_num_threads(target_threads)
        torch.set_num_interop_threads(1)
        print(f"Using CPU with {target_threads} torch threads")

    return device


def get_device_for_rank(device_preference: str, local_rank: int) -> torch.device:
    """Get the per-rank device and set the CUDA device when needed."""
    preference = (device_preference or "auto").lower()
    if preference in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.set_float32_matmul_precision("high")
    else:
        device = get_device(device_preference)
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
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build a OneCycleLR scheduler for faster, smoother convergence."""
    total_steps = max(1, steps_per_epoch * config.num_epochs)
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
        cycle_momentum=False,
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
