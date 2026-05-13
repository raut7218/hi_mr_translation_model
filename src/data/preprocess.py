"""
Data preprocessing for Hindi-Marathi NMT.

Handles: unicode normalization, whitespace cleanup, length filtering,
and train/validation splitting.
"""

from __future__ import annotations

import os
import random
import unicodedata
from pathlib import Path

from src.config import NMTConfig


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Apply NFC unicode normalization and clean whitespace."""
    # NFC normalization — canonical decomposition followed by canonical composition
    # Important for Devanagari to ensure consistent character representation
    text = unicodedata.normalize("NFC", text)
    # Collapse multiple spaces into one, strip leading/trailing
    text = " ".join(text.split())
    return text


def clean_parallel_lines(
    hi_lines: list[str],
    mr_lines: list[str],
    min_words: int = 1,
    max_words: int = 200,
) -> tuple[list[str], list[str]]:
    """Clean and filter parallel sentence pairs.

    Filters:
      - Empty lines on either side
      - Lines outside [min_words, max_words] range (either side)

    Returns:
        Tuple of (cleaned_hi, cleaned_mr) with same length.
    """
    clean_hi, clean_mr = [], []

    for hi, mr in zip(hi_lines, mr_lines):
        hi = normalize_text(hi)
        mr = normalize_text(mr)

        # Skip empty pairs
        if not hi or not mr:
            continue

        # Word-level length filtering
        hi_len = len(hi.split())
        mr_len = len(mr.split())
        if hi_len < min_words or mr_len < min_words:
            continue
        if hi_len > max_words or mr_len > max_words:
            continue

        clean_hi.append(hi)
        clean_mr.append(mr)

    return clean_hi, clean_mr


# ---------------------------------------------------------------------------
# Train / Validation split
# ---------------------------------------------------------------------------

def train_val_split(
    hi_lines: list[str],
    mr_lines: list[str],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split parallel data into train and validation sets.

    Returns:
        (train_hi, train_mr, val_hi, val_mr)
    """
    assert len(hi_lines) == len(mr_lines), "Parallel data must have same length"

    indices = list(range(len(hi_lines)))
    random.seed(seed)
    random.shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    val_indices = set(indices[:val_size])

    train_hi, train_mr = [], []
    val_hi, val_mr = [], []

    for i in range(len(hi_lines)):
        if i in val_indices:
            val_hi.append(hi_lines[i])
            val_mr.append(mr_lines[i])
        else:
            train_hi.append(hi_lines[i])
            train_mr.append(mr_lines[i])

    return train_hi, train_mr, val_hi, val_mr


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_lines(path: str | Path) -> list[str]:
    """Read a text file and return non-empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(lines: list[str], path: str | Path) -> None:
    """Write lines to a text file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(config: NMTConfig) -> dict[str, list[str]]:
    """Full preprocessing pipeline.

    1. Load raw parallel data
    2. Clean and filter
    3. Split into train / val
    4. Save processed files

    Returns:
        Dict with keys: train_hi, train_mr, val_hi, val_mr, test_hi, test_mr
    """
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # Load raw data
    print(f"Loading training data from {config.data.train_hi} ...")
    hi_lines = read_lines(config.data.train_hi)
    mr_lines = read_lines(config.data.train_mr)
    print(f"  Raw: {len(hi_lines)} Hindi, {len(mr_lines)} Marathi lines")
    assert len(hi_lines) == len(mr_lines), "Parallel data length mismatch!"

    # Clean and filter
    print("Cleaning and filtering ...")
    hi_lines, mr_lines = clean_parallel_lines(hi_lines, mr_lines)
    print(f"  After cleaning: {len(hi_lines)} parallel pairs")

    # Train / val split
    print(f"Splitting: {1 - config.data.val_split:.0%} train, {config.data.val_split:.0%} val ...")
    train_hi, train_mr, val_hi, val_mr = train_val_split(
        hi_lines, mr_lines,
        val_ratio=config.data.val_split,
        seed=config.training.seed,
    )
    print(f"  Train: {len(train_hi)} pairs")
    print(f"  Val:   {len(val_hi)} pairs")

    # Load test data
    test_hi = read_lines(config.data.test_hi)
    test_mr = read_lines(config.data.test_mr)
    test_hi = [normalize_text(line) for line in test_hi]
    test_mr = [normalize_text(line) for line in test_mr]
    print(f"  Test:  {len(test_hi)} pairs")

    # Save processed files
    out_dir = config.data.processed_dir
    write_lines(train_hi, os.path.join(out_dir, "train.hi"))
    write_lines(train_mr, os.path.join(out_dir, "train.mr"))
    write_lines(val_hi, os.path.join(out_dir, "val.hi"))
    write_lines(val_mr, os.path.join(out_dir, "val.mr"))
    write_lines(test_hi, os.path.join(out_dir, "test.hi"))
    write_lines(test_mr, os.path.join(out_dir, "test.mr"))
    print(f"  Saved processed data to {out_dir}/")

    return {
        "train_hi": train_hi, "train_mr": train_mr,
        "val_hi": val_hi, "val_mr": val_mr,
        "test_hi": test_hi, "test_mr": test_mr,
    }
