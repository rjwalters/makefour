"""
Training Callbacks

Provides callback classes for logging, checkpointing, and early stopping
during training.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current state of training, passed to callbacks."""

    epoch: int
    global_step: int
    train_loss: float
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_loss: float | None = None
    val_metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    epoch_time: float = 0.0


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, model: Module, **kwargs: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, model: Module, state: TrainingState, **kwargs: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, model: Module, state: TrainingState, **kwargs: Any) -> bool:
        """
        Called at the end of each epoch.

        Returns:
            True to continue training, False to stop early.
        """
        return True

    def on_batch_end(
        self, batch_idx: int, loss: float, global_step: int, **kwargs: Any
    ) -> None:
        """Called at the end of each training batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, model: Module, **kwargs: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(model, **kwargs)

    def on_train_end(self, model: Module, state: TrainingState, **kwargs: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_end(model, state, **kwargs)

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, model: Module, state: TrainingState, **kwargs: Any) -> bool:
        """Returns False if any callback returns False (stop training)."""
        continue_training = True
        for callback in self.callbacks:
            if not callback.on_epoch_end(model, state, **kwargs):
                continue_training = False
        return continue_training

    def on_batch_end(
        self, batch_idx: int, loss: float, global_step: int, **kwargs: Any
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, global_step, **kwargs)


class ConsoleLogger(Callback):
    """Logs training progress to console."""

    def __init__(self, log_every: int = 100):
        """
        Args:
            log_every: Log every N batches.
        """
        self.log_every = log_every
        self.epoch_start_time = 0.0

    def on_train_begin(self, model: Module, **kwargs: Any) -> None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        self.epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1} starting...")

    def on_batch_end(
        self, batch_idx: int, loss: float, global_step: int, **kwargs: Any
    ) -> None:
        if (batch_idx + 1) % self.log_every == 0:
            lr = kwargs.get("learning_rate", 0)
            logger.info(
                f"  Step {global_step}: loss={loss:.4f}, lr={lr:.2e}"
            )

    def on_epoch_end(self, model: Module, state: TrainingState, **kwargs: Any) -> bool:
        epoch_time = time.time() - self.epoch_start_time
        msg = (
            f"Epoch {state.epoch + 1} completed in {epoch_time:.1f}s: "
            f"train_loss={state.train_loss:.4f}"
        )

        if state.val_loss is not None:
            msg += f", val_loss={state.val_loss:.4f}"

        if "policy_acc" in state.val_metrics:
            msg += f", val_policy_acc={state.val_metrics['policy_acc']:.2%}"

        if "value_mae" in state.val_metrics:
            msg += f", val_value_mae={state.val_metrics['value_mae']:.4f}"

        logger.info(msg)
        return True


class FileLogger(Callback):
    """Logs training metrics to a JSON file."""

    def __init__(self, log_path: str | Path):
        """
        Args:
            log_path: Path to the log file.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    def on_epoch_end(self, model: Module, state: TrainingState, **kwargs: Any) -> bool:
        entry = {
            "epoch": state.epoch + 1,
            "global_step": state.global_step,
            "train_loss": state.train_loss,
            "train_metrics": state.train_metrics,
            "val_loss": state.val_loss,
            "val_metrics": state.val_metrics,
            "learning_rate": state.learning_rate,
            "epoch_time": state.epoch_time,
        }
        self.history.append(entry)

        # Write full history (overwrite each time for safety)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return True

    def on_train_end(self, model: Module, state: TrainingState, **kwargs: Any) -> None:
        logger.info(f"Training log saved to {self.log_path}")


class Checkpoint(Callback):
    """Saves model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_every: int = 5,
        save_best: bool = True,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints.
            save_every: Save checkpoint every N epochs.
            save_best: Whether to save the best model separately.
            metric: Metric to track for best model.
            mode: "min" or "max" for the metric.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.save_best = save_best
        self.metric = metric
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value
        else:
            return current > self.best_value

    def _get_metric_value(self, state: TrainingState) -> float | None:
        if self.metric == "val_loss":
            return state.val_loss
        elif self.metric == "train_loss":
            return state.train_loss
        elif self.metric in state.val_metrics:
            return state.val_metrics[self.metric]
        elif self.metric in state.train_metrics:
            return state.train_metrics[self.metric]
        return None

    def on_epoch_end(self, model: Module, state: TrainingState, **kwargs: Any) -> bool:
        optimizer = kwargs.get("optimizer")
        config = kwargs.get("config", {})

        # Save periodic checkpoint
        if (state.epoch + 1) % self.save_every == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{state.epoch + 1}.pt"
            self._save_checkpoint(path, model, optimizer, state, config)
            logger.info(f"Checkpoint saved: {path}")

        # Save best checkpoint
        if self.save_best:
            current_value = self._get_metric_value(state)
            if current_value is not None and self._is_better(current_value):
                self.best_value = current_value
                self.best_epoch = state.epoch
                path = self.checkpoint_dir / "best_model.pt"
                self._save_checkpoint(path, model, optimizer, state, config)
                logger.info(
                    f"New best model saved: {self.metric}={current_value:.4f} "
                    f"(epoch {state.epoch + 1})"
                )

        return True

    def _save_checkpoint(
        self,
        path: Path,
        model: Module,
        optimizer: Optimizer | None,
        state: TrainingState,
        config: dict,
    ) -> None:
        checkpoint = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model_state_dict": model.state_dict(),
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
            "train_metrics": state.train_metrics,
            "val_metrics": state.val_metrics,
            "config": config,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def on_train_end(self, model: Module, state: TrainingState, **kwargs: Any) -> None:
        optimizer = kwargs.get("optimizer")
        config = kwargs.get("config", {})

        # Save final checkpoint
        path = self.checkpoint_dir / "final_model.pt"
        self._save_checkpoint(path, model, optimizer, state, config)
        logger.info(f"Final checkpoint saved: {path}")

        if self.save_best and self.best_epoch >= 0:
            logger.info(
                f"Best model was at epoch {self.best_epoch + 1} "
                f"with {self.metric}={self.best_value:.4f}"
            )


class EarlyStopping(Callback):
    """Stops training when a metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        """
        Args:
            patience: Number of epochs with no improvement before stopping.
            metric: Metric to track.
            mode: "min" or "max" for the metric.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.epochs_without_improvement = 0

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta

    def _get_metric_value(self, state: TrainingState) -> float | None:
        if self.metric == "val_loss":
            return state.val_loss
        elif self.metric == "train_loss":
            return state.train_loss
        elif self.metric in state.val_metrics:
            return state.val_metrics[self.metric]
        elif self.metric in state.train_metrics:
            return state.train_metrics[self.metric]
        return None

    def on_epoch_end(self, model: Module, state: TrainingState, **kwargs: Any) -> bool:
        current_value = self._get_metric_value(state)

        if current_value is None:
            logger.warning(f"Early stopping metric '{self.metric}' not found")
            return True

        if self._is_better(current_value):
            self.best_value = current_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            logger.info(
                f"No improvement in {self.metric} for {self.epochs_without_improvement} epochs "
                f"(best: {self.best_value:.4f}, current: {current_value:.4f})"
            )

        if self.epochs_without_improvement >= self.patience:
            logger.info(
                f"Early stopping triggered after {self.patience} epochs without improvement"
            )
            return False

        return True


class GradientLogger(Callback):
    """Logs gradient statistics for debugging."""

    def __init__(self, log_every: int = 100):
        """
        Args:
            log_every: Log every N batches.
        """
        self.log_every = log_every

    def on_batch_end(
        self, batch_idx: int, loss: float, global_step: int, **kwargs: Any
    ) -> None:
        if (batch_idx + 1) % self.log_every == 0:
            model = kwargs.get("model")
            if model is not None:
                grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        grad_norms.append(grad_norm)

                if grad_norms:
                    avg_norm = sum(grad_norms) / len(grad_norms)
                    max_norm = max(grad_norms)
                    logger.debug(
                        f"  Gradients: avg_norm={avg_norm:.4f}, max_norm={max_norm:.4f}"
                    )


def load_checkpoint(
    path: str | Path,
    model: Module,
    optimizer: Optimizer | None = None,
    strict: bool = True,
) -> dict:
    """
    Loads a checkpoint and restores model/optimizer state.

    Args:
        path: Path to checkpoint file.
        model: Model to restore.
        optimizer: Optional optimizer to restore.
        strict: Whether to strictly enforce state dict matching.

    Returns:
        The checkpoint dictionary with metadata (epoch, metrics, etc.).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint
