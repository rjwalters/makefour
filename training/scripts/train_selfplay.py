#!/usr/bin/env python3
"""
Self-Play Training Script

Implements AlphaZero-style training loop:
1. Generate games through self-play with current model
2. Train model on new data
3. Evaluate against reference opponents
4. Repeat

Usage:
    python scripts/train_selfplay.py --iterations 10 --games-per-iteration 500
"""

import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import (
    ConnectFourDataset,
    save_games_jsonl,
    load_games_jsonl,
    encode_flat_binary,
)
from src.self_play import SelfPlayWorker
from src.training import Trainer, TrainerConfig, ConsoleLogger, Checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SimpleMLP(nn.Module):
    """MLP architecture matching the training script."""

    def __init__(self, input_size=85, hidden_sizes=None, dropout=0.1):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128]

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_size, 7)
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        shared = self.shared(x)
        policy = self.policy_head(shared)
        value = self.value_head(shared)
        return policy, value


class NeuralModel:
    """Wrapper to make PyTorch model compatible with SelfPlayWorker."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, board, to_move):
        """Predict policy and value for a position."""
        with torch.no_grad():
            # Encode position
            encoded = encode_flat_binary(board, to_move)
            tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)

            # Forward pass
            policy_logits, value = self.model(tensor)

            # Convert to numpy and ensure Python native types for JSON serialization
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            policy = [float(p) for p in policy]  # Convert to Python floats
            value = float(value.cpu().numpy()[0, 0])

            return policy, value


def evaluate_vs_random(model: nn.Module, device: str, num_games: int = 50) -> float:
    """Quick evaluation against random opponent."""
    from src.evaluation import Arena, RandomAgent, NeuralAgent
    import tempfile
    import os

    # Export model temporarily - move to CPU for export
    model_cpu = model.to("cpu")
    model_cpu.eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = f.name

    dummy_input = torch.randn(1, 85)
    torch.onnx.export(
        model_cpu,
        dummy_input,
        temp_path,
        input_names=["board"],
        output_names=["policy", "value"],
        opset_version=14,
    )
    # Move model back to original device
    model.to(device)

    try:
        # Create agents
        neural_agent = NeuralAgent(temp_path, temperature=0.0)
        random_agent = RandomAgent()

        # Play match
        arena = Arena({"neural": neural_agent, "random": random_agent})
        result = arena.run_match("neural", "random", num_games=num_games)

        # Return win rate
        return result.agent1_wins / num_games
    finally:
        os.unlink(temp_path)


def train_iteration(
    model: nn.Module,
    games: list,
    device: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.001,
) -> dict:
    """Train model on generated games."""
    # Create dataset from games list
    dataset = ConnectFourDataset(games=games, encoding="flat-binary", augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            boards = batch["board"].to(device)
            move_targets = batch["move"].to(device)
            value_targets = batch["value"].to(device)

            # Forward
            policy_logits, value = model(boards)

            # Compute losses
            policy_loss = nn.functional.cross_entropy(policy_logits, move_targets)
            value_loss = nn.functional.mse_loss(value.squeeze(), value_targets)
            loss = policy_loss + 0.5 * value_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1

        total_loss += epoch_loss / len(loader)

    return {
        "avg_loss": total_loss / epochs,
        "avg_policy_loss": total_policy_loss / total_batches,
        "avg_value_loss": total_value_loss / total_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-play training loop")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--games-per-iteration", type=int, default=500, help="Games to generate per iteration")
    parser.add_argument("--epochs-per-iteration", type=int, default=10, help="Training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Self-play temperature")
    parser.add_argument("--output-dir", type=str, default="checkpoints/selfplay", help="Output directory")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create or load model (using [128, 64] to match existing checkpoints)
    model = SimpleMLP(input_size=85, hidden_sizes=[128, 64], dropout=0.1)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training history
    history = []

    # Initial evaluation
    logger.info("Initial evaluation vs random...")
    win_rate = evaluate_vs_random(model, device)
    logger.info(f"Initial win rate vs random: {win_rate:.1%}")

    # Training loop
    for iteration in range(1, args.iterations + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"ITERATION {iteration}/{args.iterations}")
        logger.info(f"{'='*50}")

        # Generate self-play games
        logger.info(f"Generating {args.games_per_iteration} self-play games...")
        start_time = time.time()

        neural_model = NeuralModel(model, device)
        worker = SelfPlayWorker(
            model=neural_model,
            temperature=args.temperature,
            temperature_threshold=15,
            add_noise=True,
        )
        games = worker.play_games(args.games_per_iteration)

        gen_time = time.time() - start_time
        logger.info(f"Generated {len(games)} games in {gen_time:.1f}s")

        # Calculate game statistics
        total_positions = sum(len(g.positions) for g in games)
        p1_wins = sum(1 for g in games if g.metadata.get("result") == "player1_win")
        p2_wins = sum(1 for g in games if g.metadata.get("result") == "player2_win")
        draws = sum(1 for g in games if g.metadata.get("result") == "draw")
        logger.info(f"Positions: {total_positions}, P1 wins: {p1_wins}, P2 wins: {p2_wins}, Draws: {draws}")

        # Save games
        games_path = output_dir / f"games_iter_{iteration:03d}.jsonl"
        save_games_jsonl(games, str(games_path))

        # Train on new data
        logger.info(f"Training for {args.epochs_per_iteration} epochs...")
        start_time = time.time()

        metrics = train_iteration(
            model=model,
            games=games,
            device=device,
            epochs=args.epochs_per_iteration,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.1f}s")
        logger.info(f"Loss: {metrics['avg_loss']:.4f} (policy: {metrics['avg_policy_loss']:.4f}, value: {metrics['avg_value_loss']:.4f})")

        # Evaluate
        logger.info("Evaluating vs random...")
        win_rate = evaluate_vs_random(model, device)
        logger.info(f"Win rate vs random: {win_rate:.1%}")

        # Save checkpoint
        checkpoint_path = output_dir / f"model_iter_{iteration:03d}.pt"
        torch.save({
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "win_rate_vs_random": win_rate,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Record history
        history.append({
            "iteration": iteration,
            "games": len(games),
            "positions": total_positions,
            "loss": metrics["avg_loss"],
            "policy_loss": metrics["avg_policy_loss"],
            "value_loss": metrics["avg_value_loss"],
            "win_rate_vs_random": win_rate,
            "gen_time": gen_time,
            "train_time": train_time,
        })

    # Save final model and history
    final_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
    }, final_path)
    logger.info(f"\nFinal model saved: {final_path}")

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    for h in history:
        logger.info(f"Iter {h['iteration']}: loss={h['loss']:.4f}, win_rate={h['win_rate_vs_random']:.1%}")


if __name__ == "__main__":
    main()
