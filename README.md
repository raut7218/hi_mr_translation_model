# Hindi - Marathi Neural Machine Translation

LSTM Seq2Seq + Bahdanau Attention model for Hindi to Marathi translation, built with PyTorch.

This repository is configured for local Windows CPU training.

## Project Structure

```
configs/
    default.yaml            # Local CPU profile
src/
    config.py               # Config loader (YAML to dataclasses)
    data/
        preprocess.py         # Text cleaning, filtering, train/val split
        tokenizer.py          # Shared BPE tokenizer (SentencePiece)
        dataset.py            # PyTorch Dataset + DataLoader
    model/
        attention.py          # Bahdanau (additive) attention
        encoder.py            # Bidirectional LSTM encoder
        decoder.py            # LSTM decoder with attention
        seq2seq.py            # Seq2Seq wrapper + model factory
    training/
        trainer.py            # Training loop with optional MLflow tracking
        utils.py              # Seed, device, optimizer, checkpoint utils
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
    train.hi / train.mr     # 241K sentence pairs
    test.hi / test.mr       # 10.4K sentence pairs
outputs/                  # Generated at runtime
    tokenizer/              # SentencePiece model
    processed/              # Cleaned/split data
    checkpoints/            # Model checkpoints
    plots/                  # Training curves
    mlruns/                 # MLflow experiment logs
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

The training script will also bootstrap preprocessing and tokenizer training automatically if the processed files or tokenizer model are missing.

## Notes for CPU Training

- The project uses a CPU-only runtime path.
- DataLoader worker processes are disabled for Windows compatibility.
- The default config is tuned to be practical on a local machine.
- MLflow tracking is optional and disabled by default.

## Model Architecture

- Encoder: Bidirectional LSTM
- Decoder: LSTM with Bahdanau attention
- Tokenizer: Shared BPE vocabulary for Hindi and Marathi
- Attention: Additive attention with learned alignment

## Evaluation Metrics

All scores use a 0-100 scale:
- BLEU-100: corpus-level BLEU via sacrebleu
- CHRF++-100: character n-gram F-score with word bigrams via sacrebleu

## Outputs

Generated artifacts are written under outputs/:
- outputs/tokenizer/ for the SentencePiece model
- outputs/processed/ for cleaned and split datasets
- outputs/checkpoints/ for checkpoints
- outputs/plots/ for training curves
- outputs/mlruns/ for optional MLflow logs

## LLM Usage Disclosure

Claude (Anthropic) was used for code assistance during development. All architectural and design decisions were independently reasoned and verified.
