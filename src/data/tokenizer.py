"""
Shared BPE tokenizer using SentencePiece.

Trains a single shared subword vocabulary on concatenated Hindi + Marathi text.
Both languages share Devanagari script, making shared BPE highly effective.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm

from src.config import TokenizerConfig


class SharedTokenizer:
    """Wrapper around a SentencePiece BPE model for shared Hi-Mr vocabulary."""

    def __init__(self, model_path: Optional[str] = None):
        self.sp: Optional[spm.SentencePieceProcessor] = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: list[str],
        config: TokenizerConfig,
    ) -> None:
        """Train a SentencePiece BPE model on the given texts.

        Args:
            texts: All training sentences (Hindi + Marathi concatenated).
            config: Tokenizer config with vocab_size, model_prefix, etc.
        """
        # Write combined text to a temp file for SentencePiece training
        model_dir = os.path.dirname(config.model_prefix)
        os.makedirs(model_dir, exist_ok=True)

        train_file = config.model_prefix + "_train_corpus.txt"
        with open(train_file, "w", encoding="utf-8") as f:
            for line in texts:
                f.write(line.strip() + "\n")

        print(f"Training SentencePiece BPE (vocab_size={config.vocab_size}) ...")
        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=config.model_prefix,
            vocab_size=config.vocab_size,
            model_type="bpe",
            character_coverage=config.character_coverage,
            pad_id=config.pad_id,
            unk_id=config.unk_id,
            bos_id=config.bos_id,
            eos_id=config.eos_id,
            # Treat whitespace as a normal character for Devanagari
            split_digits=True,
            byte_fallback=True,
            num_threads=os.cpu_count() or 4,
        )
        print(f"  Saved model to {config.model_prefix}.model")

        # Load the freshly trained model
        self.load(config.model_prefix + ".model")

        # Clean up temp file
        if os.path.exists(train_file):
            os.remove(train_file)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a trained SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        print(f"  Loaded tokenizer: vocab_size={self.sp.get_piece_size()}")

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Tokenize text to token IDs."""
        assert self.sp is not None, "Tokenizer not loaded. Call train() or load() first."
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        assert self.sp is not None, "Tokenizer not loaded."
        # Filter out special tokens before decoding
        special = {self.pad_id, self.bos_id, self.eos_id}
        ids = [i for i in ids if i not in special]
        return self.sp.decode(ids)

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[list[int]]:
        """Tokenize a batch of texts."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        assert self.sp is not None
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        assert self.sp is not None
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        assert self.sp is not None
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        assert self.sp is not None
        return self.sp.eos_id()

    @property
    def unk_id(self) -> int:
        assert self.sp is not None
        return self.sp.unk_id()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_tokenizer(config: TokenizerConfig, texts: Optional[list[str]] = None) -> SharedTokenizer:
    """Build or load a tokenizer.

    If the model file exists, load it. Otherwise, train a new one.
    """
    model_path = config.model_prefix + ".model"
    tokenizer = SharedTokenizer()

    if os.path.exists(model_path):
        print(f"Loading existing tokenizer from {model_path}")
        tokenizer.load(model_path)
    else:
        if texts is None:
            raise ValueError(
                f"Tokenizer model not found at {model_path} and no training texts provided."
            )
        tokenizer.train(texts, config)

    return tokenizer
