"""
Training Infrastructure for Connect Four Neural Networks

This module provides the core training loop, loss functions, and optimization
utilities for training neural network bots.
"""

from .callbacks import (
    Callback,
    CallbackList,
    Checkpoint,
    ConsoleLogger,
    EarlyStopping,
    FileLogger,
    GradientLogger,
    TrainingState,
    load_checkpoint,
)
from .losses import (
    combined_loss,
    entropy_loss,
    policy_accuracy,
    policy_loss,
    policy_top_k_accuracy,
    value_loss,
    value_mae,
)
from .optimizers import (
    WarmupCosineScheduler,
    create_optimizer,
    create_scheduler,
    get_current_lr,
    set_lr,
)
from .trainer import Trainer, TrainerConfig

__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    # Loss functions
    "policy_loss",
    "value_loss",
    "combined_loss",
    "entropy_loss",
    # Metrics
    "policy_accuracy",
    "policy_top_k_accuracy",
    "value_mae",
    # Optimizers
    "create_optimizer",
    "create_scheduler",
    "WarmupCosineScheduler",
    "get_current_lr",
    "set_lr",
    # Callbacks
    "Callback",
    "CallbackList",
    "TrainingState",
    "ConsoleLogger",
    "FileLogger",
    "Checkpoint",
    "EarlyStopping",
    "GradientLogger",
    "load_checkpoint",
]
