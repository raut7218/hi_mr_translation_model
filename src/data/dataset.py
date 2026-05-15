"""
PyTorch Dataset and DataLoader for parallel translation data.

Handles tokenization, padding, and batching for efficient GPU training.
"""

from __future__ import annotations

import math
import random
from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.config import NMTConfig
from src.data.tokenizer import SharedTokenizer


class TranslationDataset(Dataset):
    """Dataset for parallel sentence pairs (already tokenized to IDs)."""

    def __init__(
        self,
        src_ids: list[list[int]],
        tgt_ids: list[list[int]],
        max_seq_len: int = 128,
    ):
        """
        Args:
            src_ids: List of source token ID sequences.
            tgt_ids: List of target token ID sequences.
            max_seq_len: Maximum sequence length (truncate longer sequences).
        """
        assert len(src_ids) == len(tgt_ids), "Source and target must have same length"
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        src = self.src_ids[idx][: self.max_seq_len]
        tgt = self.tgt_ids[idx][: self.max_seq_len]
        return src, tgt


class BucketBatchSampler(Sampler[list[int]]):
    """Yield batches of indices with similar sequence lengths."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_multiplier: int = 50,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def _bucketize(self) -> list[list[int]]:
        indices = sorted(
            range(len(self.lengths)),
            key=lambda idx: (self.lengths[idx], idx),
        )
        return [
            indices[i : i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]

    def __iter__(self):
        buckets = self._bucketize()
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(buckets)
            self._epoch += 1

        for bucket in buckets:
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for bucket in self._bucketize():
            if self.drop_last:
                total += len(bucket) // self.batch_size
            else:
                total += math.ceil(len(bucket) / self.batch_size)
        return total


def collate_fn(
    batch: list[tuple[list[int], list[int]]],
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate function with dynamic padding.

    Pads all sequences in the batch to the length of the longest sequence.

    Returns:
        Dict with keys:
          - src: (batch, max_src_len) LongTensor
          - tgt: (batch, max_tgt_len) LongTensor
          - src_lengths: (batch,) LongTensor — actual lengths before padding
    """
    src_seqs, tgt_seqs = zip(*batch)

    # Compute max lengths in this batch
    src_lengths = [len(s) for s in src_seqs]
    tgt_lengths = [len(t) for t in tgt_seqs]
    max_src = max(src_lengths)
    max_tgt = max(tgt_lengths)

    # Pad sequences
    src_padded = [s + [pad_id] * (max_src - len(s)) for s in src_seqs]
    tgt_padded = [t + [pad_id] * (max_tgt - len(t)) for t in tgt_seqs]

    return {
        "src": torch.tensor(src_padded, dtype=torch.long),
        "tgt": torch.tensor(tgt_padded, dtype=torch.long),
        "src_lengths": torch.tensor(src_lengths, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _tokenize_pairs(
    src_lines: list[str],
    tgt_lines: list[str],
    tokenizer: SharedTokenizer,
    max_seq_len: int,
    min_seq_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Tokenize parallel lines and filter by token length."""
    src_ids_all, tgt_ids_all = [], []

    for src, tgt in zip(src_lines, tgt_lines):
        s = tokenizer.encode(src)
        t = tokenizer.encode(tgt)
        # Filter by token-level length
        if len(s) < min_seq_len or len(t) < min_seq_len:
            continue
        if len(s) > max_seq_len or len(t) > max_seq_len:
            continue
        src_ids_all.append(s)
        tgt_ids_all.append(t)

    return src_ids_all, tgt_ids_all


def get_dataloaders(
    config: NMTConfig,
    tokenizer: SharedTokenizer,
    train_src: list[str],
    train_tgt: list[str],
    val_src: list[str],
    val_tgt: list[str],
    test_src: Optional[list[str]] = None,
    test_tgt: Optional[list[str]] = None,
) -> dict[str, DataLoader]:
    """Build DataLoaders for train, val, and optionally test splits.

    Returns:
        Dict with "train", "val", and optionally "test" DataLoaders.
    """
    pad_id = tokenizer.pad_id

    # Tokenize
    print("Tokenizing training data ...")
    tr_src, tr_tgt = _tokenize_pairs(
        train_src, train_tgt, tokenizer,
        config.data.max_seq_len, config.data.min_seq_len,
    )
    print(f"  Train pairs after token-length filtering: {len(tr_src)}")

    print("Tokenizing validation data ...")
    va_src, va_tgt = _tokenize_pairs(
        val_src, val_tgt, tokenizer,
        config.data.max_seq_len, config.data.min_seq_len,
    )
    print(f"  Val pairs after token-length filtering: {len(va_src)}")

    # Datasets
    train_ds = TranslationDataset(tr_src, tr_tgt, config.data.max_seq_len)
    val_ds = TranslationDataset(va_src, va_tgt, config.data.max_seq_len)
    train_lengths = [max(len(src), len(tgt)) for src, tgt in zip(tr_src, tr_tgt)]

    _collate = partial(collate_fn, pad_id=pad_id)
    num_workers = max(0, int(config.training.num_workers))
    train_batch_sampler = BucketBatchSampler(
        train_lengths,
        batch_size=config.training.batch_size,
        bucket_multiplier=50,
        shuffle=True,
        drop_last=True,
        seed=config.training.seed,
    )

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=False,
        ),
    }

    if test_src and test_tgt:
        print("Tokenizing test data ...")
        te_src, te_tgt = _tokenize_pairs(
            test_src, test_tgt, tokenizer,
            max_seq_len=512,  # Don't filter test data aggressively
            min_seq_len=1,
        )
        print(f"  Test pairs: {len(te_src)}")
        test_ds = TranslationDataset(te_src, te_tgt, max_seq_len=512)
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=False,
        )

    return loaders
