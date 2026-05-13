"""
Evaluation entry point.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt

Steps:
  1. Load config, tokenizer, and test data
  2. Load trained model from checkpoint
  3. Run inference on test set
  4. Compute and print BLEU-100 and CHRF++-100
  5. Save translations to file
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from src.config import parse_args, ensure_dirs
from src.data.dataset import get_dataloaders
from src.data.preprocess import read_lines
from src.data.tokenizer import build_tokenizer
from src.evaluation.inference import translate_batch
from src.evaluation.metrics import compute_all_metrics
from src.model.seq2seq import build_model
from src.training.utils import get_device, load_checkpoint


def main() -> None:
    config = parse_args()
    ensure_dirs(config)

    device = get_device()

    # Load tokenizer
    tokenizer = build_tokenizer(config.tokenizer)

    # Load test data
    processed_dir = config.data.processed_dir
    test_hi = read_lines(os.path.join(processed_dir, "test.hi"))
    test_mr = read_lines(os.path.join(processed_dir, "test.mr"))
    print(f"Test set: {len(test_hi)} sentence pairs")

    # Build and load model
    model = build_model(config.model, tokenizer)
    checkpoint_path = getattr(config, "_checkpoint", None)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.training.checkpoint_dir, "best.pt")
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    # Run inference
    print("\n" + "=" * 60)
    print(f"EVALUATING (strategy={config.decoding.strategy})")
    print("=" * 60)

    all_translations: list[str] = []

    # Process in batches
    batch_size = config.training.batch_size
    for i in tqdm(range(0, len(test_hi), batch_size), desc="Translating"):
        batch_hi = test_hi[i : i + batch_size]

        # Tokenize
        src_ids = tokenizer.encode_batch(batch_hi)
        max_len = max(len(s) for s in src_ids)
        src_padded = [s + [tokenizer.pad_id] * (max_len - len(s)) for s in src_ids]

        src = torch.tensor(src_padded, dtype=torch.long, device=device)
        src_lengths = torch.tensor([len(s) for s in src_ids], dtype=torch.long, device=device)

        translations = translate_batch(
            model, src, src_lengths, tokenizer,
            strategy=config.decoding.strategy,
            beam_size=config.decoding.beam_size,
            max_len=config.decoding.max_decode_len,
            length_penalty=config.decoding.length_penalty,
        )
        all_translations.extend(translations)

    # Compute metrics
    metrics = compute_all_metrics(all_translations, test_mr)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  BLEU-100:   {metrics['bleu_100']:.2f}")
    print(f"  CHRF++-100: {metrics['chrf_100']:.2f}")
    print(f"  Test pairs: {len(test_mr)}")

    # Save translations
    output_file = os.path.join(config.plotting.output_dir, "test_translations.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for hi, ref, hyp in zip(test_hi, test_mr, all_translations):
            f.write(f"SRC: {hi}\n")
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n")
            f.write("\n")
    print(f"  Translations saved to: {output_file}")

    # Print sample translations
    print("\n" + "=" * 60)
    print("SAMPLE TRANSLATIONS")
    print("=" * 60)
    for i in range(min(10, len(all_translations))):
        print(f"\n--- Example {i + 1} ---")
        print(f"  SRC: {test_hi[i]}")
        print(f"  REF: {test_mr[i]}")
        print(f"  HYP: {all_translations[i]}")


if __name__ == "__main__":
    main()
