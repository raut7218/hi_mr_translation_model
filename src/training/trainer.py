"""
Trainer: main training loop with MLflow tracking.

Handles:
  - Training with teacher forcing (with decay)
  - Periodic validation (loss + BLEU + CHRF++)
  - MLflow logging of all metrics and hyperparameters
  - Checkpoint saving (best + periodic)
  - LR scheduling
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import NMTConfig
from src.data.tokenizer import SharedTokenizer
from src.evaluation.inference import translate_batch
from src.evaluation.metrics import compute_all_metrics
from src.model.seq2seq import Seq2Seq
from src.training.utils import (
    get_optimizer,
    get_scheduler,
    save_checkpoint,
)
from src.visualization.plots import plot_training_curves


class Trainer:
    """Training loop for Seq2Seq NMT model."""

    def __init__(
        self,
        model: Seq2Seq,
        config: NMTConfig,
        tokenizer: SharedTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer & scheduler
        self.optimizer = get_optimizer(model, config.training)
        self.scheduler = get_scheduler(
            self.optimizer,
            config.training,
            steps_per_epoch=max(1, len(train_loader)),
        )

        # Loss function: cross-entropy with label smoothing, ignoring PAD
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=config.training.label_smoothing,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_bleu": [],
            "val_bleu": [],
            "train_chrf": [],
            "val_chrf": [],
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            Training history dict.
        """
        cfg = self.config.training
        mlflow_enabled = self.config.mlflow.enabled

        mlflow_ctx = nullcontext()
        if mlflow_enabled:
            # Setup MLflow only when explicitly enabled.
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            mlflow.set_experiment(self.config.mlflow.experiment_name)
            mlflow_ctx = mlflow.start_run()

        with mlflow_ctx:
            # Log hyperparameters
            if mlflow_enabled:
                self._log_hyperparams()

            for epoch in range(1, cfg.num_epochs + 1):
                # Compute teacher forcing ratio (decays over epochs)
                tf_ratio = max(
                    0.0,
                    cfg.teacher_forcing_ratio - cfg.teacher_forcing_decay * (epoch - 1),
                )

                # --- Train ---
                train_loss = self._train_epoch(epoch, tf_ratio)
                self.history["train_loss"].append(train_loss)

                # --- Validate ---
                if epoch % cfg.eval_every_n_epochs == 0:
                    val_loss, val_metrics = self._validate_epoch(epoch)
                    self.history["val_loss"].append(val_loss)
                    self.history["val_bleu"].append(val_metrics["bleu_100"])
                    self.history["val_chrf"].append(val_metrics["chrf_100"])

                    # LR scheduling
                    # Log to MLflow
                    if mlflow_enabled:
                        mlflow.log_metrics(
                            {
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "val_bleu_100": val_metrics["bleu_100"],
                                "val_chrf_100": val_metrics["chrf_100"],
                                "teacher_forcing_ratio": tf_ratio,
                                "learning_rate": self.optimizer.param_groups[0]["lr"],
                            },
                            step=epoch,
                        )

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            self.model, self.optimizer, epoch,
                            {"val_loss": val_loss, **val_metrics},
                            os.path.join(cfg.checkpoint_dir, "best.pt"),
                        )

                    print(
                        f"  Epoch {epoch}/{cfg.num_epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"BLEU: {val_metrics['bleu_100']:.2f} | "
                        f"CHRF++: {val_metrics['chrf_100']:.2f} | "
                        f"TF: {tf_ratio:.2f} | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    if mlflow_enabled:
                        mlflow.log_metric("train_loss", train_loss, step=epoch)

                # Periodic checkpoint
                if epoch % cfg.save_every_n_epochs == 0:
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        {"train_loss": train_loss},
                        os.path.join(cfg.checkpoint_dir, f"epoch_{epoch}.pt"),
                    )

            # Final plots
            plot_paths = plot_training_curves(
                self.history, self.config.plotting.output_dir,
            )
            if mlflow_enabled:
                for path in plot_paths:
                    mlflow.log_artifact(path)

        return self.history

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, tf_ratio: float) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )

        for batch in pbar:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            src_lengths = batch["src_lengths"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # outputs: (batch, tgt_len - 1, vocab_size)
            outputs = self.model(src, tgt, src_lengths, tf_ratio)

            # Compute loss
            # Reshape: (batch * (tgt_len - 1), vocab_size) vs (batch * (tgt_len - 1),)
            output_dim = outputs.shape[-1]
            outputs_flat = outputs.contiguous().view(-1, output_dim)
            targets_flat = tgt[:, 1:].contiguous().view(-1)  # Skip BOS

            loss = self.criterion(outputs_flat, targets_flat)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip,
            )
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_epoch(
        self, epoch: int, max_eval_batches: int = 20,
    ) -> tuple[float, dict[str, float]]:
        """Validate: compute loss and translation metrics on a subset.

        Args:
            epoch: Current epoch number.
            max_eval_batches: Max batches for metric computation (to save time).

        Returns:
            (val_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_hypotheses: list[str] = []
        all_references: list[str] = []

        for i, batch in enumerate(self.val_loader):
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            src_lengths = batch["src_lengths"].to(self.device)

            # Loss computation (with teacher forcing = 1.0 for consistent loss)
            outputs = self.model(src, tgt, src_lengths, teacher_forcing_ratio=1.0)
            output_dim = outputs.shape[-1]
            outputs_flat = outputs.contiguous().view(-1, output_dim)
            targets_flat = tgt[:, 1:].contiguous().view(-1)
            loss = self.criterion(outputs_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

            # Translation metrics (on a subset to save time)
            if i < max_eval_batches:
                translations = translate_batch(
                    self.model, src, src_lengths, self.tokenizer,
                    strategy="greedy",
                    max_len=self.config.decoding.max_decode_len,
                )
                references = [
                    self.tokenizer.decode(tgt[j].tolist())
                    for j in range(tgt.size(0))
                ]
                all_hypotheses.extend(translations)
                all_references.extend(references)

        val_loss = total_loss / max(num_batches, 1)

        # Compute metrics
        if all_hypotheses:
            metrics = compute_all_metrics(all_hypotheses, all_references)
        else:
            metrics = {"bleu_100": 0.0, "chrf_100": 0.0}

        return val_loss, metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_hyperparams(self) -> None:
        """Log all config hyperparameters to MLflow."""
        cfg = self.config
        params = {
            "model.embedding_dim": cfg.model.embedding_dim,
            "model.hidden_dim": cfg.model.hidden_dim,
            "model.num_layers": cfg.model.num_layers,
            "model.dropout": cfg.model.dropout,
            "model.attention_type": cfg.model.attention_type,
            "model.bidirectional": cfg.model.bidirectional_encoder,
            "model.embedding_type": cfg.model.embedding_type,
            "model.tie_embeddings": cfg.model.tie_embeddings,
            "training.batch_size": cfg.training.batch_size,
            "training.learning_rate": cfg.training.learning_rate,
            "training.optimizer": cfg.training.optimizer,
            "training.grad_clip": cfg.training.grad_clip,
            "training.label_smoothing": cfg.training.label_smoothing,
            "training.teacher_forcing_ratio": cfg.training.teacher_forcing_ratio,
            "training.teacher_forcing_decay": cfg.training.teacher_forcing_decay,
            "tokenizer.vocab_size": cfg.tokenizer.vocab_size,
            "tokenizer.type": cfg.tokenizer.type,
            "data.max_seq_len": cfg.data.max_seq_len,
            "data.val_split": cfg.data.val_split,
            "decoding.strategy": cfg.decoding.strategy,
        }
        mlflow.log_params(params)
