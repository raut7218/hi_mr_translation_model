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
import csv
import json
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import mlflow
except ImportError:
    mlflow = None

from src.config import NMTConfig
from src.data.tokenizer import SharedTokenizer
from src.evaluation.inference import translate_batch
from src.evaluation.metrics import compute_all_metrics
from src.model.seq2seq import Seq2Seq
from src.training.utils import (
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    is_distributed,
    is_main_process,
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
        self.use_amp = bool(
            config.training.mixed_precision and device.type == "cuda"
        )
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Loss function: cross-entropy with label smoothing, ignoring PAD
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=config.training.label_smoothing,
            reduction="sum",
        )
        self.microbatch_size = self._select_microbatch_size(config.training.batch_size)

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

        if self.use_amp:
            print("Mixed precision: enabled (FP16 AMP)")
        else:
            print("Mixed precision: disabled")
        if self.microbatch_size < config.training.batch_size:
            print(
                "CUDA microbatching: enabled "
                f"({self.microbatch_size} samples at a time, "
                f"effective batch {config.training.batch_size})"
            )

        if is_distributed():
            device_ids = [device.index] if device.type == "cuda" else None
            self.model = DDP(self.model, device_ids=device_ids)

        # Optimizer & scheduler
        self.optimizer = get_optimizer(self.model, config.training)
        self.scheduler = get_scheduler(
            self.optimizer,
            config.training,
            steps_per_epoch=max(1, len(train_loader)),
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            Training history dict.
        """
        cfg = self.config.training
        mlflow_enabled = self.config.mlflow.enabled and is_main_process()

        mlflow_ctx = nullcontext()
        if mlflow_enabled:
            if mlflow is None:
                raise ImportError(
                    "MLflow tracking is enabled, but mlflow is not installed. "
                    "Install requirements.txt or set mlflow.enabled=false."
                )
            # Setup MLflow only when explicitly enabled.
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            mlflow.set_experiment(self.config.mlflow.experiment_name)
            mlflow_ctx = mlflow.start_run()

        with mlflow_ctx:
            # Log hyperparameters
            if mlflow_enabled:
                self._log_hyperparams()

            for epoch in range(1, cfg.num_epochs + 1):
                sampler = getattr(self.train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

                # Compute teacher forcing ratio (decays over epochs)
                tf_ratio = max(
                    0.0,
                    cfg.teacher_forcing_ratio - cfg.teacher_forcing_decay * (epoch - 1),
                )

                # --- Train ---
                total_loss, num_batches = self._train_epoch(epoch, tf_ratio)
                if is_distributed():
                    loss_tensor = torch.tensor(
                        [total_loss, float(num_batches)],
                        device=self.device,
                    )
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    train_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1.0)
                else:
                    train_loss = total_loss / max(num_batches, 1)
                self.history["train_loss"].append(train_loss)

                # --- Validate ---
                if epoch % cfg.eval_every_n_epochs == 0 and is_main_process():
                    train_metrics = self._compute_translation_metrics(
                        self.train_loader,
                        max_eval_batches=cfg.train_metric_batches,
                    )
                    val_loss, val_metrics = self._validate_epoch(
                        epoch,
                        max_eval_batches=cfg.val_metric_batches,
                    )
                    self.history["val_loss"].append(val_loss)
                    self.history["train_bleu"].append(train_metrics["bleu_100"])
                    self.history["val_bleu"].append(val_metrics["bleu_100"])
                    self.history["train_chrf"].append(train_metrics["chrf_100"])
                    self.history["val_chrf"].append(val_metrics["chrf_100"])

                    # LR scheduling
                    # Log to MLflow
                    if mlflow_enabled:
                        mlflow.log_metrics(
                            {
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "train_bleu_100": train_metrics["bleu_100"],
                                "val_bleu_100": val_metrics["bleu_100"],
                                "train_chrf_100": train_metrics["chrf_100"],
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
                            self._unwrap_model(), self.optimizer, epoch,
                            {"val_loss": val_loss, **val_metrics},
                            os.path.join(cfg.checkpoint_dir, "best.pt"),
                        )

                    print(
                        f"  Epoch {epoch}/{cfg.num_epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Train BLEU: {train_metrics['bleu_100']:.2f} | "
                        f"Val BLEU: {val_metrics['bleu_100']:.2f} | "
                        f"Train CHRF++: {train_metrics['chrf_100']:.2f} | "
                        f"Val CHRF++: {val_metrics['chrf_100']:.2f} | "
                        f"TF: {tf_ratio:.2f} | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    if mlflow_enabled:
                        mlflow.log_metric("train_loss", train_loss, step=epoch)

                # Periodic checkpoint
                if epoch % cfg.save_every_n_epochs == 0 and is_main_process():
                    save_checkpoint(
                        self._unwrap_model(), self.optimizer, epoch,
                        {"train_loss": train_loss},
                        os.path.join(cfg.checkpoint_dir, f"epoch_{epoch}.pt"),
                    )

            # Final plots
            if is_main_process():
                plot_paths = plot_training_curves(
                    self.history, self.config.plotting.output_dir,
                )
                history_paths = self._save_history()
                if mlflow_enabled:
                    for path in plot_paths:
                        mlflow.log_artifact(path)
                    for path in history_paths:
                        mlflow.log_artifact(path)

        return self.history

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, tf_ratio: float) -> tuple[float, int]:
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
            disable=not is_main_process(),
        )

        for batch in pbar:
            src = batch["src"].to(self.device, non_blocking=True)
            tgt = batch["tgt"].to(self.device, non_blocking=True)
            src_lengths = batch["src_lengths"].to(self.device, non_blocking=True)

            batch_loss = self._train_batch_with_microbatches(src, tgt, src_lengths, tf_ratio)

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += batch_loss
            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        return total_loss, num_batches

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
            src = batch["src"].to(self.device, non_blocking=True)
            tgt = batch["tgt"].to(self.device, non_blocking=True)
            src_lengths = batch["src_lengths"].to(self.device, non_blocking=True)

            # Loss computation (with teacher forcing = 1.0 for consistent loss)
            loss = self._eval_batch_loss(src, tgt, src_lengths)

            total_loss += loss
            num_batches += 1

            # Translation metrics (on a subset to save time)
            if i < max_eval_batches:
                translations = self._translate_in_microbatches(src, src_lengths)
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

    @torch.no_grad()
    def _compute_translation_metrics(
        self,
        loader: DataLoader,
        max_eval_batches: int,
    ) -> dict[str, float]:
        """Compute sampled BLEU/CHRF++ for a loader."""
        if max_eval_batches <= 0:
            return {"bleu_100": 0.0, "chrf_100": 0.0}

        self.model.eval()
        all_hypotheses: list[str] = []
        all_references: list[str] = []

        for i, batch in enumerate(loader):
            if i >= max_eval_batches:
                break
            src = batch["src"].to(self.device, non_blocking=True)
            tgt = batch["tgt"].to(self.device, non_blocking=True)
            src_lengths = batch["src_lengths"].to(self.device, non_blocking=True)

            translations = self._translate_in_microbatches(src, src_lengths)
            references = [
                self.tokenizer.decode(tgt[j].tolist())
                for j in range(tgt.size(0))
            ]
            all_hypotheses.extend(translations)
            all_references.extend(references)

        if not all_hypotheses:
            return {"bleu_100": 0.0, "chrf_100": 0.0}
        return compute_all_metrics(all_hypotheses, all_references)

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

    def _autocast(self):
        """Return the right autocast context for the active device."""
        if not self.use_amp:
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    def _select_microbatch_size(self, batch_size: int) -> int:
        """Choose an internal chunk size for small local CUDA cards."""
        env_value = os.environ.get("HI_MR_MICROBATCH_SIZE")
        if env_value:
            try:
                return max(1, min(batch_size, int(env_value)))
            except ValueError:
                print(f"Ignoring invalid HI_MR_MICROBATCH_SIZE={env_value!r}")

        if self.device.type != "cuda":
            return batch_size

        props = torch.cuda.get_device_properties(self.device)
        total_gb = props.total_memory / (1024 ** 3)
        if total_gb <= 4.5:
            return min(batch_size, 8)
        if total_gb <= 8.5:
            return min(batch_size, 16)
        return batch_size

    def _iter_microbatches(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor | None,
        src_lengths: torch.Tensor,
    ):
        step = max(1, self.microbatch_size)
        for start in range(0, src.size(0), step):
            end = start + step
            yield (
                src[start:end],
                tgt[start:end] if tgt is not None else None,
                src_lengths[start:end],
            )

    def _loss_sum_and_tokens(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(src, tgt, src_lengths, teacher_forcing_ratio)
        output_dim = outputs.shape[-1]
        outputs_flat = outputs.contiguous().view(-1, output_dim)
        targets_flat = tgt[:, 1:].contiguous().view(-1)
        loss_sum = self.criterion(outputs_flat, targets_flat)
        token_count = targets_flat.ne(self.tokenizer.pad_id).sum().clamp_min(1)
        return loss_sum, token_count

    def _train_batch_with_microbatches(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
        tf_ratio: float,
    ) -> float:
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            try:
                with torch.no_grad():
                    total_tokens = tgt[:, 1:].ne(self.tokenizer.pad_id).sum().clamp_min(1)

                total_loss_sum = 0.0
                for src_mb, tgt_mb, src_lengths_mb in self._iter_microbatches(
                    src, tgt, src_lengths,
                ):
                    assert tgt_mb is not None
                    with self._autocast():
                        loss_sum, _ = self._loss_sum_and_tokens(
                            src_mb, tgt_mb, src_lengths_mb, tf_ratio,
                        )
                        loss = loss_sum / total_tokens
                    self.scaler.scale(loss).backward()
                    total_loss_sum += float(loss_sum.detach().item())

                return total_loss_sum / float(total_tokens.item())
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc) or self.microbatch_size <= 1:
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                self.microbatch_size = max(1, self.microbatch_size // 2)
                torch.cuda.empty_cache()
                print(
                    "CUDA OOM while training; retrying batch with "
                    f"microbatch_size={self.microbatch_size}"
                )

    @torch.no_grad()
    def _eval_batch_loss(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> float:
        total_loss_sum = 0.0
        total_tokens = 0
        for src_mb, tgt_mb, src_lengths_mb in self._iter_microbatches(
            src, tgt, src_lengths,
        ):
            assert tgt_mb is not None
            with self._autocast():
                loss_sum, token_count = self._loss_sum_and_tokens(
                    src_mb, tgt_mb, src_lengths_mb, teacher_forcing_ratio=1.0,
                )
            total_loss_sum += float(loss_sum.item())
            total_tokens += int(token_count.item())
        return total_loss_sum / max(total_tokens, 1)

    @torch.no_grad()
    def _translate_in_microbatches(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> list[str]:
        translations: list[str] = []
        for src_mb, _, src_lengths_mb in self._iter_microbatches(src, None, src_lengths):
            with self._autocast():
                translations.extend(
                    translate_batch(
                        self._unwrap_model(), src_mb, src_lengths_mb, self.tokenizer,
                        strategy="greedy",
                        max_len=self.config.decoding.max_decode_len,
                    )
                )
        return translations

    def _is_cuda_oom(self, exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "cuda" in message and "out of memory" in message

    def _save_history(self) -> list[str]:
        """Persist metric history for reports and reproducibility."""
        output_dir = self.config.plotting.output_dir
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "training_history.json")
        csv_path = os.path.join(output_dir, "training_history.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        max_len = max((len(values) for values in self.history.values()), default=0)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            keys = list(self.history.keys())
            writer.writerow(["index", *keys])
            for i in range(max_len):
                writer.writerow([
                    i + 1,
                    *[
                        self.history[key][i] if i < len(self.history[key]) else ""
                        for key in keys
                    ],
                ])

        print(f"  History saved: {json_path}")
        print(f"  History saved: {csv_path}")
        return [json_path, csv_path]

    def _unwrap_model(self) -> nn.Module:
        """Return the underlying model (DDP-safe) for checkpointing."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
