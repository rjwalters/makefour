#!/usr/bin/env python3
"""
Training Script for Connect Four Neural Networks

Usage:
    python -m scripts.train --config configs/train/supervised-mlp.yaml --data ./data/games.jsonl

Or if installed:
    train --config configs/train/supervised-mlp.yaml --data ./data/games.jsonl
"""

import logging
import sys
from pathlib import Path

import click
import torch
import yaml
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import ConnectFourDataset, load_games_jsonl, create_train_val_test_split
from src.training import (
    Trainer,
    TrainerConfig,
    Checkpoint,
    ConsoleLogger,
    EarlyStopping,
    FileLogger,
    create_optimizer,
    create_scheduler,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(model_config: dict, input_size: int, device: str) -> torch.nn.Module:
    """
    Create a model based on configuration.

    This is a placeholder that creates simple MLP/CNN architectures.
    In practice, you'd import your model classes here.
    """
    model_type = model_config.get("type", "mlp-tiny")

    if model_type == "mlp-tiny":
        return SimpleMLP(
            input_size=input_size,
            hidden_sizes=[128, 64],
            dropout=model_config.get("dropout", 0.1),
        ).to(device)
    elif model_type == "mlp-small":
        return SimpleMLP(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            dropout=model_config.get("dropout", 0.1),
        ).to(device)
    elif model_type == "mlp-medium":
        return SimpleMLP(
            input_size=input_size,
            hidden_sizes=[512, 256, 128],
            dropout=model_config.get("dropout", 0.1),
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class SimpleMLP(torch.nn.Module):
    """
    Simple MLP architecture with policy and value heads.

    This is a basic architecture for initial testing. Production models
    would use more sophisticated architectures (ResNets, Transformers, etc.)
    """

    def __init__(
        self,
        input_size: int = 85,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_sizes = hidden_sizes or [256, 128]

        # Build shared layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        self.shared = torch.nn.Sequential(*layers)

        # Policy head (7 columns)
        self.policy_head = torch.nn.Linear(prev_size, 7)

        # Value head (single scalar)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(prev_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_size).

        Returns:
            Tuple of (policy_logits, value) where:
            - policy_logits: (batch, 7) raw logits for each column
            - value: (batch, 1) position evaluation in [-1, 1]
        """
        # Flatten if needed (e.g., from 3D CNN input)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        shared = self.shared(x)
        policy = self.policy_head(shared)
        value = self.value_head(shared)
        return policy, value


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to training configuration YAML file.",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to training data (JSONL file or directory).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./checkpoints",
    help="Output directory for checkpoints and logs.",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Resume training from checkpoint.",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Override number of epochs from config.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Override batch size from config.",
)
@click.option(
    "--lr",
    type=float,
    default=None,
    help="Override learning rate from config.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device to train on.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
def main(
    config: str,
    data: str,
    output: str,
    resume: str | None,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    device: str,
    seed: int | None,
    verbose: bool,
) -> None:
    """
    Train a Connect Four neural network.

    Example:
        python -m scripts.train -c configs/train/supervised-mlp.yaml -d ./data/games.jsonl
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(config)

    # Apply command-line overrides
    training_cfg = cfg.get("training", {})
    if epochs is not None:
        training_cfg["epochs"] = epochs
    if batch_size is not None:
        training_cfg["batch_size"] = batch_size
    if lr is not None:
        training_cfg["learning_rate"] = lr
    if seed is not None:
        training_cfg["seed"] = seed

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Set random seed
    actual_seed = training_cfg.get("seed", 42)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
    logger.info(f"Random seed: {actual_seed}")

    # Load data
    logger.info(f"Loading data from {data}")
    data_path = Path(data)
    data_cfg = cfg.get("data", {})
    encoding = data_cfg.get("encoding", "flat-binary")
    augment = data_cfg.get("augment", True)
    train_ratio = data_cfg.get("train_ratio", 0.8)
    val_ratio = data_cfg.get("val_ratio", 0.1)

    games = load_games_jsonl(data_path)
    logger.info(f"Loaded {len(games)} games")

    # Split data
    train_games, val_games, test_games = create_train_val_test_split(
        games,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=actual_seed,
    )
    logger.info(
        f"Split: {len(train_games)} train, {len(val_games)} val, {len(test_games)} test"
    )

    # Create datasets
    train_dataset = ConnectFourDataset(games=train_games, encoding=encoding, augment=augment)
    val_dataset = ConnectFourDataset(games=val_games, encoding=encoding, augment=False)

    logger.info(f"Training positions: {len(train_dataset)}")
    logger.info(f"Validation positions: {len(val_dataset)}")

    # Create data loaders
    actual_batch_size = training_cfg.get("batch_size", 256)
    num_workers = training_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    # Get input size from encoding
    sample = train_dataset[0]
    input_size = sample["board"].numel()
    logger.info(f"Input size: {input_size} (encoding: {encoding})")

    # Create model
    model_cfg = cfg.get("model", {"type": "mlp-tiny"})
    model = create_model(model_cfg, input_size, device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_cfg.get('type', 'mlp-tiny')} ({total_params:,} parameters)")

    # Create optimizer
    optimizer_cfg = cfg.get("optimizer", {})
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type=optimizer_cfg.get("type", "adamw"),
        learning_rate=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )

    # Create scheduler
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler = None
    if scheduler_cfg.get("type", "none") != "none":
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=scheduler_cfg.get("type", "cosine"),
            epochs=training_cfg.get("epochs", 100),
            warmup_epochs=scheduler_cfg.get("warmup_epochs", 0),
            min_lr=scheduler_cfg.get("min_lr", 1e-6),
            steps_per_epoch=len(train_loader),
        )

    # Create trainer config
    trainer_config = TrainerConfig(
        epochs=training_cfg.get("epochs", 100),
        batch_size=actual_batch_size,
        grad_clip=training_cfg.get("grad_clip", 1.0),
        policy_weight=training_cfg.get("policy_weight", 1.0),
        value_weight=training_cfg.get("value_weight", 1.0),
        entropy_weight=training_cfg.get("entropy_weight", 0.0),
        log_every=training_cfg.get("log_every", 100),
        device=device,
        seed=actual_seed,
        num_workers=num_workers,
    )

    # Create callbacks
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ConsoleLogger(log_every=trainer_config.log_every),
        FileLogger(output_dir / "training_log.json"),
        Checkpoint(
            checkpoint_dir=output_dir,
            save_every=training_cfg.get("save_every", 5),
            save_best=True,
            metric=training_cfg.get("early_stopping", {}).get("metric", "val_loss"),
            mode=training_cfg.get("early_stopping", {}).get("mode", "min"),
        ),
    ]

    # Add early stopping if configured
    early_stopping_cfg = training_cfg.get("early_stopping", {})
    if early_stopping_cfg.get("patience"):
        callbacks.append(
            EarlyStopping(
                patience=early_stopping_cfg["patience"],
                metric=early_stopping_cfg.get("metric", "val_loss"),
                mode=early_stopping_cfg.get("mode", "min"),
            )
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_config,
        callbacks=callbacks,
    )

    # Resume from checkpoint if specified
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)

    # Train
    logger.info("Starting training...")
    logger.info(f"Epochs: {trainer_config.epochs}")
    logger.info(f"Batch size: {actual_batch_size}")
    logger.info(f"Learning rate: {training_cfg.get('learning_rate', 1e-3)}")

    results = trainer.train(train_loader, val_loader)

    # Print final results
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Final epoch: {results['final_epoch']}")
    logger.info(f"Final train loss: {results['final_train_loss']:.4f}")
    if results["final_val_loss"] is not None:
        logger.info(f"Final val loss: {results['final_val_loss']:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
