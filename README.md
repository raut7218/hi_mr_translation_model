# Hindi ↔ Marathi Neural Machine Translation

LSTM Seq2Seq + Bahdanau Attention model for Hindi → Marathi translation, built with PyTorch.

**Part I** of the [Adivaani Hiring Assignment](https://github.com/raut7218/hi_mr_translation_model) — MISN Lab, IIT Delhi.

## Project Structure

```
├── configs/
│   ├── default.yaml            # Full-size/default profile
│   └── colab.yaml              # Lean profile for Colab / T4
├── src/
│   ├── config.py               # Config loader (YAML → dataclasses)
│   ├── data/
│   │   ├── preprocess.py       # Text cleaning, filtering, train/val split
│   │   ├── tokenizer.py        # Shared BPE tokenizer (SentencePiece)
│   │   └── dataset.py          # PyTorch Dataset + DataLoader
│   ├── model/
│   │   ├── attention.py        # Bahdanau (additive) attention
│   │   ├── encoder.py          # Bidirectional LSTM encoder
│   │   ├── decoder.py          # LSTM decoder with attention
│   │   └── seq2seq.py          # Seq2Seq wrapper + model factory
│   ├── training/
│   │   ├── trainer.py          # Training loop with MLflow tracking
│   │   └── utils.py            # Seed, optimizer, checkpoint utils
│   ├── evaluation/
│   │   ├── metrics.py          # BLEU-100, CHRF++-100 (sacrebleu)
│   │   ├── inference.py        # Greedy + beam search decoding
│   │   └── translate.py        # Interactive translation CLI
│   └── visualization/
│       └── plots.py            # Loss/BLEU/CHRF++ curves
├── scripts/
│   ├── preprocess.py           # Entry: data cleaning + tokenizer training
│   ├── train.py                # Entry: model training
│   └── evaluate.py             # Entry: test set evaluation
├── data/                       # Raw parallel corpus
│   ├── train.hi / train.mr     # 241K sentence pairs
│   └── test.hi  / test.mr      # 10.4K sentence pairs
└── outputs/                    # Generated at runtime
    ├── tokenizer/              # SentencePiece model
    ├── processed/              # Cleaned/split data
    ├── checkpoints/            # Model checkpoints
    ├── plots/                  # Training curves
    └── mlruns/                 # MLflow experiment logs
```

## Setup (Colab / T4)

```bash
!git clone https://github.com/raut7218/hi_mr_translation_model.git
%cd hi_mr_translation_model

# 1. Install dependencies with uv if available
!python -m pip install --upgrade pip
!python -m pip install uv

# 2. Install project dependencies
!uv pip install -r requirements.txt

# 3. Train from a fresh clone
!python scripts/train.py
```

If `uv` is unavailable in your environment, fall back to:

```bash
!python -m pip install -r requirements.txt
!python scripts/train.py
```

The first training run bootstraps preprocessing and tokenizer training automatically if the generated artifacts are missing.

## Setup (Local Development)

```bash
# 1. Clone the repo
git clone https://github.com/raut7218/hi_mr_translation_model.git
cd hi_mr_translation_model

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Preprocess and train
python scripts/train.py
```

## Running Locally (for development)

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

python scripts/preprocess.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

For a lighter Colab-style run locally, use:

```bash
python scripts/train.py --config configs/colab.yaml
```

## Experiment Configuration

The default profile lives in `configs/default.yaml`. For Colab or a smaller T4 run, use `configs/colab.yaml`. To run an ablation:

```bash
# Copy config
cp configs/default.yaml configs/big_model.yaml
# Edit the copy (e.g., change hidden_dim to 2048)
# Train with the new config
python scripts/train.py --config configs/big_model.yaml
```

**MLflow** tracks runs in the default profile. View experiment results:
```bash
mlflow ui --backend-store-uri outputs/mlruns
```

## Model Architecture

- **Encoder**: Bidirectional 2-layer LSTM (1024 hidden, 512 embedding)
- **Decoder**: 2-layer LSTM with Bahdanau attention
- **Tokenizer**: Shared 32K BPE vocabulary (SentencePiece) for both Hindi and Marathi
- **Attention**: Additive (Bahdanau) attention with learned alignment

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Shared BPE vocab | Hindi & Marathi share Devanagari script — maximizes subword overlap |
| Tied embeddings | Encoder/decoder share embedding matrix — reduces params, improves generalization |
| Teacher forcing decay | Gradually shifts from ground-truth to model predictions during training |
| Label smoothing (0.1) | Prevents overconfident predictions, improves generalization |
| MLflow tracking | Automatic logging of all metrics, hyperparams, and model artifacts |

## Evaluation Metrics

All scores on **0–100 scale** as required:
- **BLEU-100**: Corpus-level BLEU via sacrebleu
- **CHRF++-100**: Character n-gram F-score with word bigrams via sacrebleu

## Hardware

- Training: Google Colab T4 or a local CUDA GPU
- First-run preprocessing and tokenizer bootstrap are handled automatically by `scripts/train.py`

## LLM Usage Disclosure

Claude (Anthropic) was used for code assistance during development. All architectural and design decisions were independently reasoned and verified.
