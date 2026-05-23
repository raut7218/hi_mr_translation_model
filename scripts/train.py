"""
Training entry point.

Usage:
    python scripts/train.py --config configs/colab_random.yaml

Steps:
  1. Load config, tokenizer, and processed data
  2. Build model
  3. Create DataLoaders
    4. Train on GPU with optional AMP and MLflow tracking
  5. Save checkpoints and plots

The script bootstraps preprocessing and tokenizer training when the expected
artifacts are missing so it can run from a fresh local clone with one command.
"""

import sys
import os
from pathlib import Path

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp

from src.config import parse_args, ensure_dirs
from src.data.dataset import get_dataloaders
from src.data.preprocess import read_lines
from src.data.tokenizer import build_tokenizer
from src.data.preprocess import run_preprocessing
from src.model.seq2seq import build_model
from src.training.trainer import Trainer
from src.training.utils import (
    set_seed,
    get_device,
    get_device_for_rank,
    init_distributed,
    cleanup_distributed,
    is_main_process,
    barrier,
)


def _load_training_artifacts(config, allow_bootstrap: bool = True):
    """Load or bootstrap processed data and tokenizer."""
    processed_dir = Path(config.data.processed_dir)
    required_files = [
        processed_dir / "train.hi",
        processed_dir / "train.mr",
        processed_dir / "val.hi",
        processed_dir / "val.mr",
    ]
    tokenizer_model = Path(f"{config.tokenizer.model_prefix}.model")

    data = None
    if not all(path.exists() for path in required_files):
        if not allow_bootstrap:
            raise RuntimeError(
                "Processed data missing. Run preprocessing on rank 0 first."
            )
        print("\nProcessed data not found. Running preprocessing bootstrap ...")
        data = run_preprocessing(config)

    if data is None:
        train_hi = read_lines(processed_dir / "train.hi")
        train_mr = read_lines(processed_dir / "train.mr")
        val_hi = read_lines(processed_dir / "val.hi")
        val_mr = read_lines(processed_dir / "val.mr")
    else:
        train_hi = data["train_hi"]
        train_mr = data["train_mr"]
        val_hi = data["val_hi"]
        val_mr = data["val_mr"]

    if tokenizer_model.exists():
        tokenizer = build_tokenizer(config.tokenizer)
    else:
        if not allow_bootstrap:
            raise RuntimeError(
                "Tokenizer missing. Run preprocessing on rank 0 first."
            )
        print("\nTokenizer not found. Training shared BPE tokenizer ...")
        tokenizer = build_tokenizer(config.tokenizer, texts=train_hi + train_mr)

    return tokenizer, train_hi, train_mr, val_hi, val_mr


def _run_worker(rank: int, world_size: int, config) -> None:
    if config.training.distributed_enable and world_size > 1:
        init_method = config.training.distributed_init_method
        backend = config.training.distributed_backend
        init_distributed(rank, world_size, backend=backend, init_method=init_method)
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = get_device_for_rank(config.training.device, local_rank)
    else:
        device = get_device(config.training.device)

    set_seed(
        config.training.seed + rank,
        deterministic=config.training.deterministic,
    )
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.training.cudnn_benchmark

    if is_main_process():
        ensure_dirs(config)

    if is_main_process():
        print("\n" + "=" * 60)
        print("LOADING TOKENIZER")
        print("=" * 60)
        _load_training_artifacts(config, allow_bootstrap=True)

    barrier()

    tokenizer, train_hi, train_mr, val_hi, val_mr = _load_training_artifacts(
        config, allow_bootstrap=False,
    )

    if is_main_process():
        print(f"  Train: {len(train_hi)} pairs")
        print(f"  Val:   {len(val_hi)} pairs")

    # Build DataLoaders
    if is_main_process():
        print("\n" + "=" * 60)
        print("BUILDING DATALOADERS")
        print("=" * 60)
    loaders = get_dataloaders(
        config, tokenizer,
        train_src=train_hi, train_tgt=train_mr,
        val_src=val_hi, val_tgt=val_mr,
        distributed=config.training.distributed_enable,
        rank=rank,
        world_size=world_size,
    )

    # Build model
    if is_main_process():
        print("\n" + "=" * 60)
        print("BUILDING MODEL")
        print("=" * 60)
    model = build_model(config.model, tokenizer)

    # Train
    if is_main_process():
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
    trainer = Trainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
    )

    history = trainer.train()

    if is_main_process():
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Best val loss: {trainer.best_val_loss:.4f}")
        if history["val_bleu"]:
            print(f"  Final val BLEU-100:  {history['val_bleu'][-1]:.2f}")
            print(f"  Final val CHRF++-100: {history['val_chrf'][-1]:.2f}")

    cleanup_distributed()


def _assert_two_t4_gpus() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. This training supports only 2x T4 GPUs.")
    device_count = torch.cuda.device_count()
    if device_count != 2:
        raise RuntimeError(
            f"Expected exactly 2 GPUs, found {device_count}. "
            "This training supports only 2x T4 GPUs."
        )
    for index in range(device_count):
        name = torch.cuda.get_device_name(index)
        if "T4" not in name:
            raise RuntimeError(
                f"GPU {index} is '{name}'. This training supports only 2x T4 GPUs."
            )


def main() -> None:
    config = parse_args()

    if not config.training.distributed_enable:
        raise RuntimeError("DDP must be enabled. This training supports only 2x T4 GPUs.")

    _assert_two_t4_gpus()

    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env:
        world_size = int(world_size_env)
        if world_size != 2:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} is not supported. "
                "This training supports only 2x T4 GPUs."
            )
        rank = int(os.environ.get("RANK", "0"))
        _run_worker(rank, world_size, config)
    else:
        world_size = 2
        if config.training.distributed_init_method == "env://":
            config.training.distributed_init_method = "tcp://127.0.0.1:29500"
        mp.spawn(
            _run_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
