"""
Main Trainer Class for Connect Four Neural Networks

Provides a complete training loop with validation, checkpointing, and callbacks.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .callbacks import Callback, CallbackList, TrainingState
from .losses import (
    combined_loss,
    policy_accuracy,
    policy_top_k_accuracy,
    value_mae,
)
from .optimizers import get_current_lr

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 256
    grad_clip: float | None = 1.0

    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 1.0
    entropy_weight: float = 0.0

    # Logging
    log_every: int = 100
    eval_every: int | None = None  # Evaluate every N steps (None = end of epoch only)

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Misc
    seed: int | None = 42
    num_workers: int = 0
    pin_memory: bool = True

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


class Trainer:
    """
    Complete training loop for Connect Four neural networks.

    Supports:
    - Dual-headed networks (policy + value)
    - Gradient clipping
    - Learning rate scheduling
    - Validation during training
    - Configurable callbacks (logging, checkpointing, early stopping)
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        config: TrainerConfig | None = None,
        callbacks: list[Callback] | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The neural network to train.
            optimizer: Optimizer for parameter updates.
            scheduler: Optional learning rate scheduler.
            config: Training configuration.
            callbacks: List of callbacks for logging, checkpointing, etc.
        """
        self.config = config or TrainerConfig()
        self.model = model.to(self.config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = CallbackList(callbacks)

        self.global_step = 0
        self.start_epoch = 0

        # Set random seed for reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        logger.info(f"Trainer initialized on device: {self.config.device}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            epochs: Number of epochs (overrides config if provided).

        Returns:
            Dictionary containing final metrics and training history.
        """
        epochs = epochs or self.config.epochs
        self.callbacks.on_train_begin(self.model)

        history: list[TrainingState] = []
        final_state = None

        try:
            for epoch in range(self.start_epoch, epochs):
                self.callbacks.on_epoch_begin(epoch)
                epoch_start = time.time()

                # Training epoch
                train_loss, train_metrics = self.train_epoch(train_loader)

                # Validation
                val_loss = None
                val_metrics = {}
                if val_loader is not None:
                    val_loss, val_metrics = self.validate(val_loader)

                epoch_time = time.time() - epoch_start
                current_lr = get_current_lr(self.optimizer)

                # Create state for callbacks
                state = TrainingState(
                    epoch=epoch,
                    global_step=self.global_step,
                    train_loss=train_loss,
                    train_metrics=train_metrics,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                    epoch_time=epoch_time,
                )
                history.append(state)
                final_state = state

                # Step scheduler at end of epoch
                if self.scheduler is not None:
                    self.scheduler.step()

                # Callback at end of epoch (may trigger early stopping)
                continue_training = self.callbacks.on_epoch_end(
                    self.model,
                    state,
                    optimizer=self.optimizer,
                    config=self._get_config_dict(),
                )

                if not continue_training:
                    logger.info(f"Training stopped early at epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        # Final callback
        if final_state is not None:
            self.callbacks.on_train_end(
                self.model,
                final_state,
                optimizer=self.optimizer,
                config=self._get_config_dict(),
            )

        return {
            "final_epoch": final_state.epoch + 1 if final_state else 0,
            "final_train_loss": final_state.train_loss if final_state else None,
            "final_val_loss": final_state.val_loss if final_state else None,
            "history": [self._state_to_dict(s) for s in history],
        }

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, dict[str, float]]:
        """
        Run a single training epoch.

        Args:
            dataloader: DataLoader for training data.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Accumulate metrics
        total_policy_acc = 0.0
        total_top3_acc = 0.0
        total_value_mae = 0.0

        for batch_idx, batch in enumerate(dataloader):
            loss, metrics = self.train_step(batch)
            total_loss += loss
            num_batches += 1

            total_policy_acc += metrics.get("policy_acc", 0)
            total_top3_acc += metrics.get("policy_top3_acc", 0)
            total_value_mae += metrics.get("value_mae", 0)

            self.global_step += 1

            # Callback for batch end
            self.callbacks.on_batch_end(
                batch_idx,
                loss,
                self.global_step,
                learning_rate=get_current_lr(self.optimizer),
                model=self.model,
            )

        avg_loss = total_loss / num_batches
        avg_metrics = {
            "policy_acc": total_policy_acc / num_batches,
            "policy_top3_acc": total_top3_acc / num_batches,
            "value_mae": total_value_mae / num_batches,
        }

        return avg_loss, avg_metrics

    def train_step(self, batch: dict[str, Tensor]) -> tuple[float, dict[str, float]]:
        """
        Single training step (forward + backward + optimizer step).

        Args:
            batch: Dictionary with 'board', 'move', 'value', 'legal_mask'.

        Returns:
            Tuple of (loss_value, metrics_dict).
        """
        # Move batch to device
        board = batch["board"].to(self.config.device)
        move_target = batch["move"].to(self.config.device)
        value_target = batch["value"].to(self.config.device)
        legal_mask = batch["legal_mask"].to(self.config.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        policy_logits, value_pred = self.model(board)

        # Compute loss
        loss, loss_components = combined_loss(
            policy_logits=policy_logits,
            value_pred=value_pred,
            policy_target=move_target,
            value_target=value_target,
            policy_weight=self.config.policy_weight,
            value_weight=self.config.value_weight,
            legal_mask=legal_mask,
            entropy_weight=self.config.entropy_weight,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

        # Optimizer step
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            metrics = {
                "policy_acc": policy_accuracy(policy_logits, move_target, legal_mask),
                "policy_top3_acc": policy_top_k_accuracy(
                    policy_logits, move_target, k=3, mask=legal_mask
                ),
                "value_mae": value_mae(value_pred, value_target),
            }

        return loss.item(), metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> tuple[float, dict[str, float]]:
        """
        Run validation on a dataset.

        Args:
            dataloader: DataLoader for validation data.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Accumulate metrics
        total_policy_acc = 0.0
        total_top3_acc = 0.0
        total_value_mae = 0.0

        for batch in dataloader:
            # Move batch to device
            board = batch["board"].to(self.config.device)
            move_target = batch["move"].to(self.config.device)
            value_target = batch["value"].to(self.config.device)
            legal_mask = batch["legal_mask"].to(self.config.device)

            # Forward pass
            policy_logits, value_pred = self.model(board)

            # Compute loss
            loss, _ = combined_loss(
                policy_logits=policy_logits,
                value_pred=value_pred,
                policy_target=move_target,
                value_target=value_target,
                policy_weight=self.config.policy_weight,
                value_weight=self.config.value_weight,
                legal_mask=legal_mask,
            )

            total_loss += loss.item()
            num_batches += 1

            # Compute metrics
            total_policy_acc += policy_accuracy(policy_logits, move_target, legal_mask)
            total_top3_acc += policy_top_k_accuracy(
                policy_logits, move_target, k=3, mask=legal_mask
            )
            total_value_mae += value_mae(value_pred, value_target)

        avg_loss = total_loss / num_batches
        avg_metrics = {
            "policy_acc": total_policy_acc / num_batches,
            "policy_top3_acc": total_top3_acc / num_batches,
            "value_mae": total_value_mae / num_batches,
        }

        return avg_loss, avg_metrics

    def _get_config_dict(self) -> dict:
        """Convert config to dictionary for saving."""
        return {
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "grad_clip": self.config.grad_clip,
            "policy_weight": self.config.policy_weight,
            "value_weight": self.config.value_weight,
            "entropy_weight": self.config.entropy_weight,
            "device": self.config.device,
            "seed": self.config.seed,
        }

    def _state_to_dict(self, state: TrainingState) -> dict:
        """Convert TrainingState to dictionary."""
        return {
            "epoch": state.epoch + 1,
            "global_step": state.global_step,
            "train_loss": state.train_loss,
            "train_metrics": state.train_metrics,
            "val_loss": state.val_loss,
            "val_metrics": state.val_metrics,
            "learning_rate": state.learning_rate,
            "epoch_time": state.epoch_time,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save a checkpoint manually.

        Args:
            path: Path to save the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self._get_config_dict(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True) -> None:
        """
        Load a checkpoint.

        Args:
            path: Path to the checkpoint file.
            load_optimizer: Whether to load optimizer state.
        """
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Calculate start epoch from global step if available
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
