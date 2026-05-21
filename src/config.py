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
    max_train_examples: int = 10000
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
    device: str = "auto"
    mixed_precision: bool = True
    cudnn_benchmark: bool = True
    deterministic: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    train_metric_batches: int = 10
    val_metric_batches: int = 20


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
_SECTION_MAP = {
    "data": DataConfig,
    "tokenizer": TokenizerConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "decoding": DecodingConfig,
    "mlflow": MLflowConfig,
    "plotting": PlottingConfig,
}


def _guess_project_root(config_path: Path) -> Path:
    config_dir = config_path.parent
    if config_dir.name == "configs":
        return config_dir.parent
    return config_dir


def _resolve_rel_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_config_paths(config: NMTConfig, base_dir: Path) -> None:
    config.data.train_hi = _resolve_rel_path(config.data.train_hi, base_dir)
    config.data.train_mr = _resolve_rel_path(config.data.train_mr, base_dir)
    config.data.test_hi = _resolve_rel_path(config.data.test_hi, base_dir)
    config.data.test_mr = _resolve_rel_path(config.data.test_mr, base_dir)
    config.data.processed_dir = _resolve_rel_path(config.data.processed_dir, base_dir)
    config.tokenizer.model_prefix = _resolve_rel_path(config.tokenizer.model_prefix, base_dir)
    config.training.checkpoint_dir = _resolve_rel_path(config.training.checkpoint_dir, base_dir)
    config.plotting.output_dir = _resolve_rel_path(config.plotting.output_dir, base_dir)
    config.mlflow.tracking_uri = _resolve_rel_path(config.mlflow.tracking_uri, base_dir)


def load_config(path: str | Path) -> NMTConfig:
    """Load a YAML config file and return an NMTConfig instance."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    kwargs: dict[str, Any] = {}
    for section_name, dc_cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {})
        if section_data:
            kwargs[section_name] = dc_cls(**section_data)

    config = NMTConfig(**kwargs)
    project_root = _guess_project_root(config_path)
    _resolve_config_paths(config, project_root)
    config._project_root = str(project_root)  # type: ignore[attr-defined]
    return config


def parse_args() -> NMTConfig:
    """Parse CLI arguments and return config.

    Usage:
        python scripts/train.py --config configs/default.yaml
    """
    parser = argparse.ArgumentParser(description="Hindi-Marathi NMT")
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to YAML config file. Defaults to configs/default.yaml."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (for evaluation / resume)",
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    if config_path is None:
        config_path = "configs/default.yaml"

    config = load_config(config_path)

    # Attach checkpoint path as a runtime attribute
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        ckpt_path = Path(checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            project_root = Path(getattr(config, "_project_root", Path.cwd()))
            ckpt_path = (project_root / ckpt_path).resolve()
        config._checkpoint = str(ckpt_path)  # type: ignore[attr-defined]
    else:
        config._checkpoint = None  # type: ignore[attr-defined]
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
