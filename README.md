# Hindi - Marathi Neural Machine Translation

LSTM Seq2Seq + Bahdanau Attention model for Hindi to Marathi translation, built with PyTorch.

This repository is configured for Kaggle only and requires exactly 2x T4 GPUs.

## Kaggle 2x T4 Training (Required)

Two experiment profiles are provided:

- `configs/colab_random.yaml`: LSTM Seq2Seq with randomly initialized shared BPE embeddings.
- `configs/colab_bert.yaml`: same LSTM Seq2Seq architecture with Hindi/Marathi BERT-initialized embedding tables.

Both profiles enable CUDA, FP16 AMP, pinned-memory DataLoaders, persistent workers, length-bucketed batching (single-GPU only), OneCycleLR, label smoothing, and sampled train/validation BLEU-100 and CHRF++-100 logging. Training uses the full dataset.

## Project Structure

```
configs/
    colab_random.yaml       # Colab/T4 random embedding experiment
    colab_bert.yaml         # Colab/T4 BERT-initialized embedding experiment
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
        utils.py              # Seed, optimizer, scheduler, checkpoint utils
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

## Kaggle 2x T4 Setup

1) In Kaggle, enable a 2x T4 GPU accelerator.

2) Install requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3) Place the corpus files at:

```text
data/train.hi
data/train.mr
data/test.hi
data/test.mr
```

4) Run preprocessing and training (DDP spawns automatically):

```bash
python scripts/preprocess.py --config configs/colab_random.yaml
python scripts/train.py --config configs/colab_random.yaml
```

5) Evaluate using the best checkpoint:

```bash
python scripts/evaluate.py --config configs/colab_random.yaml --checkpoint outputs/colab_random/checkpoints/best.pt
```

Repeat with `configs/colab_bert.yaml` for the BERT-initialized experiment.

Notes:
- Training will error if it does not detect exactly two T4 GPUs.
- Total batch size scales with GPU count. If you hit OOM, lower `training.batch_size`.
- Only rank 0 logs metrics, writes checkpoints, and creates plots.

## Preprocessing Behavior

Preprocessing keeps Hindi and Marathi sentence pairs aligned at every step.

- Text is normalized with NFC and whitespace cleanup.
- Empty or over-length pairs are filtered out.
- The raw training data is split into train and validation sets.
- Training uses the full dataset (no cap on training pairs).
- Validation and test sets are kept intact.

## Model Architecture

- Encoder: Bidirectional LSTM
- Decoder: LSTM with Bahdanau attention
- Tokenizer: Shared BPE vocabulary for Hindi and Marathi
- Attention: Additive attention with learned alignment
- Embeddings: Tied encoder/decoder embeddings for the random-embedding path
- BERT path: Separate Hindi encoder and Marathi decoder embedding tables initialized from the required L3Cube BERT models

## Speed Notes for T4

- FP16 AMP, CUDA, and cuDNN benchmark are enabled.
- Pinned-memory DataLoaders with persistent workers reduce input stalls.
- Bucketed batches reduce padding and wasted compute.
- OneCycleLR and label smoothing stabilize optimization for faster convergence.

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
- `training_history.json` and `training_history.csv` beside the plots for report tables

## Notes

- Only 2x T4 GPUs are supported.
- `configs/colab_random.yaml` and `configs/colab_bert.yaml` use `device: cuda`.
- Training uses the full dataset by default (`max_train_examples: null`).
- MLflow tracking stays off unless you enable it in the config.

## LLM Usage Disclosure

Claude (Anthropic) was used for code assistance during development. All architectural and design decisions were independently reasoned and verified.
