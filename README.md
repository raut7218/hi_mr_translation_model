# Hindi - Marathi Neural Machine Translation

LSTM Seq2Seq + Bahdanau Attention model for Hindi to Marathi translation, built with PyTorch.

This repository can train on Kaggle 2x T4 GPUs or locally on a single CUDA GPU.
The local path has been verified against a Windows global Python environment with
PyTorch `2.5.1+cu121` and a 4 GB NVIDIA GeForce GTX 1650.

## Training Profiles

Two experiment profiles are provided:

- `configs/colab_random.yaml`: LSTM Seq2Seq with randomly initialized shared BPE embeddings.
- `configs/colab_bert.yaml`: same LSTM Seq2Seq architecture with Hindi/Marathi BERT-initialized embedding tables.

Both profiles enable CUDA, FP16 AMP, pinned-memory DataLoaders, persistent workers, length-bucketed batching on single-GPU runs, OneCycleLR, label smoothing, and sampled train/validation BLEU-100 and CHRF++-100 logging. Training uses the full dataset.

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

## Local Single-GPU Setup

The main local command uses the global Python environment:

```bash
python scripts/train.py --config configs/colab_random.yaml
```

On a 4 GB GTX GPU, the trainer automatically keeps the configured effective
batch size and splits each batch into smaller internal CUDA microbatches when
needed. This avoids changing the committed hyperparameters while keeping memory
use realistic for local hardware.

To verify the active Python/PyTorch/CUDA environment:

```bash
python --version
python -c "import torch, sys; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi
```

Install requirements if needed:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run preprocessing explicitly, or let `scripts/train.py` bootstrap it if the
processed files/tokenizer are missing:

```bash
python scripts/preprocess.py --config configs/colab_random.yaml
python scripts/train.py --config configs/colab_random.yaml
```

Evaluate using the best checkpoint:

```bash
python scripts/evaluate.py --config configs/colab_random.yaml --checkpoint outputs/colab_random/checkpoints/best.pt
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
The first BERT run downloads the L3Cube Hindi and Marathi BERT checkpoints via
the Hugging Face cache. After the BERT vectors are converted into this
project's BPE embedding tables, they are saved under `outputs/bert_embeddings/`
and reused on later runs.

Notes:
- On local Windows single-GPU runs, DDP is disabled automatically.
- On Linux multi-GPU runs, DDP is used when `training.distributed_enable` is true.
- On 4 GB CUDA cards, internal microbatching keeps the configured effective batch size while reducing peak memory.
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
- Small local CUDA cards use internal gradient accumulation over microbatches.

## Evaluation Metrics

All scores use a 0-100 scale:

- BLEU-100: corpus-level BLEU via sacrebleu
- CHRF++-100: character n-gram F-score with word bigrams via sacrebleu

## Results Summary

Best checkpoints are selected by lowest validation loss.

| Experiment | Best Epoch | Val Loss | Val BLEU-100 | Val CHRF++-100 |
|-----------|------------|----------|--------------|----------------|
| Random embeddings | 14 | 3.958 | 11.23 | 35.25 |
| BERT embeddings | 12 | 3.841 | 11.40 | 35.46 |

Training curves and metric logs are available under:

- `outputs/colab_random/plots/`
- `outputs/colab_bert/plots/`

## Outputs

Generated artifacts are written under `outputs/`:

- `outputs/tokenizer/` for the SentencePiece model
- `outputs/processed/` for cleaned and split datasets
- `outputs/checkpoints/` for checkpoints
- `outputs/plots/` for training curves
- `outputs/mlruns/` for optional MLflow logs
- `training_history.json` and `training_history.csv` beside the plots for report tables

## Notes

- Local single-GPU CUDA training is supported.
- `configs/colab_random.yaml` and `configs/colab_bert.yaml` use `device: cuda`.
- BERT-initialized runs require the Hugging Face BERT checkpoints once; cached
  embedding tables are reused after that first successful build.
- Training uses the full dataset by default (`max_train_examples: null`).
- MLflow tracking stays off unless you enable it in the config.

## LLM Usage Disclosure

Claude (Anthropic) was used for code assistance during development. All architectural and design decisions were independently reasoned and verified.
