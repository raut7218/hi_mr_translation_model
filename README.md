# Hindi - Marathi Neural Machine Translation

LSTM Seq2Seq + Bahdanau Attention model for Hindi to Marathi translation, built with PyTorch.

This repository is configured for local Windows CPU training.

## Current Training Setup

The default config is tuned for a small, fast test run on this machine:

- CPU-only execution
- Training subset limited to the first 10,000 examples after preprocessing
- Learning rate: `1e-3`
- Optimizer: `AdamW`
- Weight decay: `0.01`
- Scheduler: `OneCycleLR` stepped every batch
- Batch size: `16`
- DataLoader workers: `2`
- Bucketed batches by sentence length to reduce padding
- Label smoothing: `0.1`
- Gradient clipping: `5.0`
- MLflow: disabled by default

## Project Structure

```
configs/
    default.yaml            # Local CPU profile used by default
src/
    config.py               # Config loader (YAML to dataclasses)
    data/
        preprocess.py         # Text cleaning, filtering, train/val split
        tokenizer.py          # Shared BPE tokenizer (SentencePiece)
        dataset.py            # PyTorch Dataset, bucketed batching, DataLoader
    model/
        attention.py          # Bahdanau (additive) attention
        encoder.py            # Bidirectional LSTM encoder
        decoder.py            # LSTM decoder with attention
        seq2seq.py            # Seq2Seq wrapper + model factory
    training/
        trainer.py            # Training loop with validation + optional MLflow
        utils.py              # Seed, CPU threading, optimizer, scheduler, checkpoint utils
    evaluation/
        metrics.py            # BLEU-100, CHRF++-100 (sacrebleu)
        inference.py          # Greedy + beam search decoding
        translate.py          # Interactive translation CLI
    visualization/
        plots.py              # Loss/BLEU/CHRF++ curves
scripts/
    preprocess.py           # Entry: data cleaning + tokenizer training
    train.py                # Entry: model training
    evaluate.py             # Entry: test set evaluation
data/                     # Raw parallel corpus
    train.hi / train.mr     # Training pairs
    test.hi / test.mr       # Test pairs
outputs/                  # Generated at runtime
    tokenizer/              # SentencePiece model
    processed/              # Cleaned/split data
    checkpoints/            # Model checkpoints
    plots/                  # Training curves
    mlruns/                 # Optional MLflow experiment logs
```

## Setup on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell script execution is blocked, run this once in the same shell session:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

## Local Workflow

Run preprocessing, then training, then evaluation:

```powershell
python scripts/preprocess.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

`scripts/train.py` will bootstrap preprocessing and tokenizer training automatically if the processed files or tokenizer model are missing.

## Preprocessing Behavior

Preprocessing keeps Hindi and Marathi sentence pairs aligned at every step.

- Text is normalized with NFC and whitespace cleanup.
- Empty or over-length pairs are filtered out.
- The raw training data is split into train and validation sets.
- Only the first 10,000 training pairs are kept for the fast test run.
- Validation and test sets are kept intact.

## Model Architecture

- Encoder: Bidirectional LSTM
- Decoder: LSTM with Bahdanau attention
- Tokenizer: Shared BPE vocabulary for Hindi and Marathi
- Attention: Additive attention with learned alignment
- Embeddings: Tied encoder/decoder embeddings for the random-embedding path

## Why This Setup Is More Robust

The current configuration is aimed at smoother and faster convergence on CPU:

- `AdamW` adds light regularization.
- `OneCycleLR` helps the model move quickly early and settle more smoothly later.
- Label smoothing reduces overconfident predictions.
- Gradient clipping keeps updates stable.
- Bucketed batches reduce padding and improve CPU efficiency.
- Small worker parallelism speeds up data loading without fully saturating the machine.

## Evaluation Metrics

All scores use a 0-100 scale:

- BLEU-100: corpus-level BLEU via sacrebleu
- CHRF++-100: character n-gram F-score with word bigrams via sacrebleu

## Outputs

Generated artifacts are written under `outputs/`:

- `outputs/tokenizer/` for the SentencePiece model
- `outputs/processed/` for cleaned and split datasets
- `outputs/checkpoints/` for checkpoints
- `outputs/plots/` for training curves
- `outputs/mlruns/` for optional MLflow logs

## Notes

- The project runs on CPU by default.
- The train split is capped at 10k examples for the fast local run.
- The test set is not reduced.
- MLflow tracking stays off unless you enable it in the config.

## LLM Usage Disclosure

Claude (Anthropic) was used for code assistance during development. All architectural and design decisions were independently reasoned and verified.
