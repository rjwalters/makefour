"""
Optimizer Factory and Learning Rate Schedulers

Provides factory functions for creating optimizers and learning rate schedulers
from configuration dictionaries.
"""

import math
from typing import Any

import torch
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    StepLR,
)


def create_optimizer(
    params,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
    **kwargs: Any,
) -> Optimizer:
    """
    Creates an optimizer from configuration.

    Args:
        params: Model parameters to optimize.
        optimizer_type: One of "adam", "adamw", "sgd".
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        momentum: Momentum for SGD.
        betas: Beta parameters for Adam/AdamW.
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer_type is not recognized.
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        return Adam(
            params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type == "adamw":
        return AdamW(
            params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type == "sgd":
        return SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            "Supported: adam, adamw, sgd"
        )


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 100,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    steps_per_epoch: int | None = None,
    milestones: list[int] | None = None,
    gamma: float = 0.1,
    **kwargs: Any,
) -> LRScheduler | None:
    """
    Creates a learning rate scheduler from configuration.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of "cosine", "step", "multistep", "onecycle", "warmup_cosine", "none".
        epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs (for warmup_cosine).
        min_lr: Minimum learning rate (for cosine schedules).
        steps_per_epoch: Steps per epoch (required for onecycle).
        milestones: Epochs at which to decay LR (for multistep).
        gamma: LR decay factor (for step/multistep).
        **kwargs: Additional scheduler-specific arguments.

    Returns:
        Configured scheduler instance, or None if scheduler_type is "none".

    Raises:
        ValueError: If scheduler_type is not recognized.
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "none":
        return None

    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr,
            **kwargs,
        )

    elif scheduler_type == "step":
        step_size = kwargs.pop("step_size", epochs // 3)
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs,
        )

    elif scheduler_type == "multistep":
        if milestones is None:
            # Default milestones at 60% and 80% of training
            milestones = [int(epochs * 0.6), int(epochs * 0.8)]
        return MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            **kwargs,
        )

    elif scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for OneCycleLR")
        max_lr = optimizer.param_groups[0]["lr"]
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

    elif scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            min_lr=min_lr,
        )

    elif scheduler_type == "cosine_restarts":
        # Cosine annealing with warm restarts
        t_0 = kwargs.pop("t_0", epochs // 4)
        t_mult = kwargs.pop("t_mult", 2)
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=t_mult,
            eta_min=min_lr,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            "Supported: cosine, step, multistep, onecycle, warmup_cosine, cosine_restarts, none"
        )


class WarmupCosineScheduler(LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.

    This is a common schedule for training neural networks:
    - Linear warmup from 0 to base_lr over warmup_epochs
    - Cosine decay from base_lr to min_lr over remaining epochs
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of warmup epochs.
            total_epochs: Total number of training epochs.
            min_lr: Minimum learning rate after decay.
            last_epoch: The index of last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def get_current_lr(optimizer: Optimizer) -> float:
    """Gets the current learning rate from an optimizer."""
    return optimizer.param_groups[0]["lr"]


def set_lr(optimizer: Optimizer, lr: float) -> None:
    """Sets the learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
