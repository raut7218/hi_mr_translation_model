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

from src.config import parse_args, ensure_dirs
from src.data.dataset import get_dataloaders
from src.data.preprocess import read_lines
from src.data.tokenizer import build_tokenizer
from src.data.preprocess import run_preprocessing
from src.model.seq2seq import build_model
from src.training.trainer import Trainer
from src.training.utils import set_seed, get_device


def _load_training_artifacts(config):
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
        print("\nTokenizer not found. Training shared BPE tokenizer ...")
        tokenizer = build_tokenizer(config.tokenizer, texts=train_hi + train_mr)

    return tokenizer, train_hi, train_mr, val_hi, val_mr


def main() -> None:
    config = parse_args()
    ensure_dirs(config)

    # Reproducibility
    # Device
    set_seed(config.training.seed, deterministic=config.training.deterministic)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.training.cudnn_benchmark
    device = get_device(config.training.device)

    # Load tokenizer (must already be trained via preprocess.py)
    print("\n" + "=" * 60)
    print("LOADING TOKENIZER")
    print("=" * 60)
    tokenizer, train_hi, train_mr, val_hi, val_mr = _load_training_artifacts(config)

    print(f"  Train: {len(train_hi)} pairs")
    print(f"  Val:   {len(val_hi)} pairs")

    # Build DataLoaders
    print("\n" + "=" * 60)
    print("BUILDING DATALOADERS")
    print("=" * 60)
    loaders = get_dataloaders(
        config, tokenizer,
        train_src=train_hi, train_tgt=train_mr,
        val_src=val_hi, val_tgt=val_mr,
    )

    # Build model
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    model = build_model(config.model, tokenizer)

    # Train
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

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    if history["val_bleu"]:
        print(f"  Final val BLEU-100:  {history['val_bleu'][-1]:.2f}")
        print(f"  Final val CHRF++-100: {history['val_chrf'][-1]:.2f}")


if __name__ == "__main__":
    main()
