"""
Interactive / file-based translation CLI.

Usage:
    python -m src.evaluation.translate --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
    python -m src.evaluation.translate --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --input file.txt
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.config import load_config
from src.data.tokenizer import SharedTokenizer, build_tokenizer
from src.evaluation.inference import translate_batch
from src.model.seq2seq import Seq2Seq, build_model
from src.training.utils import get_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Hindi → Marathi")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, default=None, help="Input file (one sentence per line)")
    parser.add_argument("--strategy", type=str, default=None, help="greedy or beam")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.training.device)

    # Load tokenizer
    tokenizer = build_tokenizer(config.tokenizer)

    # Build and load model
    model = build_model(config.model, tokenizer)
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()

    strategy = args.strategy or config.decoding.strategy

    if args.input:
        # File-based translation
        with open(args.input, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Translating {len(lines)} sentences ...")
        for line in lines:
            translation = _translate_single(
                model, line, tokenizer, device, config, strategy,
            )
            print(f"SRC: {line}")
            print(f"TGT: {translation}")
            print()
    else:
        # Interactive mode
        print("Interactive Hindi → Marathi Translation")
        print("Type a Hindi sentence and press Enter. Type 'quit' to exit.")
        print("-" * 50)

        while True:
            try:
                line = input("Hindi> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if line.lower() in ("quit", "exit", "q"):
                break
            if not line:
                continue

            translation = _translate_single(
                model, line, tokenizer, device, config, strategy,
            )
            print(f"Marathi> {translation}\n")


def _translate_single(
    model: Seq2Seq,
    text: str,
    tokenizer: SharedTokenizer,
    device: torch.device,
    config,
    strategy: str,
) -> str:
    """Translate a single sentence."""
    ids = tokenizer.encode(text)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)

    translations = translate_batch(
        model, src, src_lengths, tokenizer,
        strategy=strategy,
        beam_size=config.decoding.beam_size,
        max_len=config.decoding.max_decode_len,
        length_penalty=config.decoding.length_penalty,
    )
    return translations[0]


if __name__ == "__main__":
    main()
