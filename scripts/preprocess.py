"""
Preprocessing entry point.

Usage:
    python scripts/preprocess.py --config configs/default.yaml

Steps:
  1. Clean and filter raw parallel data
  2. Split into train / validation
  3. Train shared BPE tokenizer on combined Hi+Mr text
  4. Save processed data and tokenizer model
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import parse_args, ensure_dirs
from src.data.preprocess import run_preprocessing
from src.data.tokenizer import build_tokenizer


def main() -> None:
    config = parse_args()
    ensure_dirs(config)

    # Step 1: Preprocess raw data
    data = run_preprocessing(config)

    # Step 2: Train shared BPE tokenizer on combined train data
    print("\n" + "=" * 60)
    print("TOKENIZER TRAINING")
    print("=" * 60)

    combined_texts = data["train_hi"] + data["train_mr"]
    print(f"Training tokenizer on {len(combined_texts)} sentences (Hi + Mr combined)")

    tokenizer = build_tokenizer(config.tokenizer, texts=combined_texts)

    # Verify tokenizer
    sample_hi = data["train_hi"][0]
    sample_mr = data["train_mr"][0]
    print(f"\n  Sample Hindi:   {sample_hi}")
    print(f"  Encoded:        {tokenizer.encode(sample_hi)[:20]}...")
    print(f"  Decoded:        {tokenizer.decode(tokenizer.encode(sample_hi))}")
    print(f"\n  Sample Marathi: {sample_mr}")
    print(f"  Encoded:        {tokenizer.encode(sample_mr)[:20]}...")
    print(f"  Decoded:        {tokenizer.decode(tokenizer.encode(sample_mr))}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
