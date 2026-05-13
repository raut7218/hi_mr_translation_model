"""
Configuration loader for Hindi-Marathi NMT.

Loads a YAML config file into typed dataclasses. Supports CLI override
via --config flag and optional --override key=value pairs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dataclasses — mirror the YAML structure
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    train_hi: str = "data/train.hi"
    train_mr: str = "data/train.mr"
    test_hi: str = "data/test.hi"
    test_mr: str = "data/test.mr"
    val_split: float = 0.05
    max_seq_len: int = 128
    min_seq_len: int = 2
    processed_dir: str = "outputs/processed"


@dataclass
class TokenizerConfig:
    type: str = "bpe"
    vocab_size: int = 32000
    model_prefix: str = "outputs/tokenizer/shared_bpe"
    character_coverage: float = 1.0
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3


@dataclass
class ModelConfig:
    embedding_dim: int = 512
    hidden_dim: int = 1024
    num_layers: int = 2
    dropout: float = 0.3
    attention_type: str = "bahdanau"
    bidirectional_encoder: bool = True
    embedding_type: str = "random"
    tie_embeddings: bool = True
    # BERT-specific (only used when embedding_type == "bert")
    bert_hi_model: str = "l3cube-pune/hindi-bert-v2"
    bert_mr_model: str = "l3cube-pune/marathi-bert-v2"
    freeze_bert_embeddings: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 30
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    label_smoothing: float = 0.1
    teacher_forcing_ratio: float = 1.0
    teacher_forcing_decay: float = 0.02
    num_workers: int = 4
    seed: int = 42
    checkpoint_dir: str = "outputs/checkpoints"
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 1


@dataclass
class DecodingConfig:
    strategy: str = "greedy"
    beam_size: int = 5
    max_decode_len: int = 150
    length_penalty: float = 0.6


@dataclass
class MLflowConfig:
    enabled: bool = True
    experiment_name: str = "hi_mr_nmt_part1"
    tracking_uri: str = "outputs/mlruns"


@dataclass
class PlottingConfig:
    output_dir: str = "outputs/plots"


@dataclass
class NMTConfig:
    """Top-level configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_SECTION_MAP = {
    "data": DataConfig,
    "tokenizer": TokenizerConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "decoding": DecodingConfig,
    "mlflow": MLflowConfig,
    "plotting": PlottingConfig,
}


def _is_colab_environment() -> bool:
    """Detect Google Colab without importing Colab-specific modules."""
    return bool(
        os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("COLAB_GPU")
        or os.environ.get("COLAB_TPU_ADDR")
    )


def load_config(path: str | Path) -> NMTConfig:
    """Load a YAML config file and return an NMTConfig instance."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    kwargs: dict[str, Any] = {}
    for section_name, dc_cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {})
        if section_data:
            kwargs[section_name] = dc_cls(**section_data)

    return NMTConfig(**kwargs)


def parse_args() -> NMTConfig:
    """Parse CLI arguments and return config.

    Usage:
        python scripts/train.py --config configs/default.yaml
    """
    parser = argparse.ArgumentParser(description="Hindi-Marathi NMT")
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to YAML config file. Defaults to configs/colab.yaml in Colab "
            "and configs/default.yaml elsewhere."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (for evaluation / resume)",
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    if config_path is None:
        config_path = "configs/colab.yaml" if _is_colab_environment() else "configs/default.yaml"

    config = load_config(config_path)

    # Attach checkpoint path as a runtime attribute
    config._checkpoint = getattr(args, "checkpoint", None)  # type: ignore[attr-defined]
    return config


def ensure_dirs(config: NMTConfig) -> None:
    """Create all output directories specified in the config."""
    dirs = [
        config.data.processed_dir,
        os.path.dirname(config.tokenizer.model_prefix),
        config.training.checkpoint_dir,
        config.plotting.output_dir,
    ]
    if config.mlflow.enabled:
        dirs.append(config.mlflow.tracking_uri)
    for d in dirs:
        os.makedirs(d, exist_ok=True)
