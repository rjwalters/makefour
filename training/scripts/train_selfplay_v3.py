#!/usr/bin/env python3
"""
Enhanced Self-Play Training Script v3

Improvements over v2:
- Larger network architecture
- Learning rate decay schedule
- Temperature annealing during self-play
- Evaluation against minimax opponents (not just random)
- Checkpointing best model based on minimax performance
- Accumulated replay buffer for diverse training data

Usage:
    python scripts/train_selfplay_v3.py --iterations 30 --games-per-iteration 1000
"""

import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import (
    ConnectFourDataset,
    save_games_jsonl,
    encode_flat_binary,
)
from src.self_play import SelfPlayWorker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class EnhancedMLP(nn.Module):
    """Larger MLP without LayerNorm (for ONNX opset 14 compatibility)."""

    def __init__(self, input_size=85, hidden_size=256, num_blocks=3, dropout=0.1):
        super().__init__()

        # Build deep network with residual-style connections
        layers = []

        # Input projection
        layers.extend([
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        # Deep hidden layers
        for _ in range(num_blocks):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        self.shared = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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
            encoded = encode_flat_binary(board, to_move)
            tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
            policy_logits, value = self.model(tensor)
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            policy = [float(p) for p in policy]
            value = float(value.cpu().numpy()[0, 0])
            return policy, value


class PyTorchAgent:
    """Agent that uses PyTorch model directly (no ONNX export needed)."""

    def __init__(self, model: nn.Module, device: str, temperature: float = 0.0):
        self.model = model
        self.device = device
        self.temperature = temperature

    def get_move(self, board, to_move):
        """Get move using the neural network."""
        self.model.eval()
        with torch.no_grad():
            encoded = encode_flat_binary(board, to_move)
            tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
            policy_logits, _ = self.model(tensor)

            # Apply temperature
            if self.temperature > 0:
                policy_logits = policy_logits / self.temperature

            probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]

            # Mask invalid moves
            for col in range(7):
                if board[0][col] is not None:
                    probs[col] = 0

            # Renormalize
            total = probs.sum()
            if total > 0:
                probs = probs / total
            else:
                # All moves invalid (shouldn't happen)
                return 0

            if self.temperature == 0:
                return int(np.argmax(probs))
            else:
                return int(np.random.choice(7, p=probs))


def play_game(agent1, agent2):
    """Play a single game between two agents. Returns 1 if agent1 wins, -1 if agent2 wins, 0 for draw."""
    from src.data.game import check_winner

    board = [[None] * 7 for _ in range(6)]
    to_move = 1  # Player 1 starts

    for _ in range(42):  # Max moves
        if to_move == 1:
            move = agent1.get_move(board, to_move)
        else:
            move = agent2.get_move(board, to_move)

        # Make move
        for row in range(5, -1, -1):
            if board[row][move] is None:
                board[row][move] = to_move
                break

        # Check winner
        winner = check_winner(board)
        if winner == 1:
            return 1
        elif winner == 2:
            return -1

        # Check draw
        if all(board[0][col] is not None for col in range(7)):
            return 0

        to_move = 3 - to_move  # Switch player

    return 0  # Draw


def evaluate_vs_minimax(model: nn.Module, device: str, depth: int = 3, num_games: int = 20) -> float:
    """Evaluate against minimax opponent using PyTorch directly."""
    from src.evaluation import MinimaxAgent

    neural_agent = PyTorchAgent(model, device, temperature=0.0)
    minimax_agent = MinimaxAgent(depth=depth)

    wins = 0
    draws = 0

    for i in range(num_games):
        # Alternate who goes first
        if i % 2 == 0:
            result = play_game(neural_agent, minimax_agent)
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
        else:
            result = play_game(minimax_agent, neural_agent)
            if result == -1:
                wins += 1
            elif result == 0:
                draws += 1

    return (wins + 0.5 * draws) / num_games


def evaluate_vs_random(model: nn.Module, device: str, num_games: int = 50) -> float:
    """Quick evaluation against random opponent using PyTorch directly."""
    from src.evaluation import RandomAgent

    neural_agent = PyTorchAgent(model, device, temperature=0.0)
    random_agent = RandomAgent()

    wins = 0

    for i in range(num_games):
        if i % 2 == 0:
            result = play_game(neural_agent, random_agent)
            if result == 1:
                wins += 1
        else:
            result = play_game(random_agent, neural_agent)
            if result == -1:
                wins += 1

    return wins / num_games


def train_iteration(
    model: nn.Module,
    games: list,
    device: str,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
) -> dict:
    """Train model on generated games."""
    # Fix: Set value_target to result (game outcome) instead of model prediction
    # This ensures the model learns from actual outcomes, not its own predictions
    for game in games:
        for pos in game.positions:
            pos.value_target = pos.result  # Use actual game outcome

    dataset = ConnectFourDataset(games=games, encoding="flat-binary", augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

            policy_logits, value = model(boards)
            policy_loss = nn.functional.cross_entropy(policy_logits, move_targets)
            value_loss = nn.functional.mse_loss(value.squeeze(), value_targets)
            loss = policy_loss + value_loss  # Equal weight

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1

        scheduler.step()
        total_loss += epoch_loss / len(loader)

    return {
        "avg_loss": total_loss / epochs,
        "avg_policy_loss": total_policy_loss / total_batches,
        "avg_value_loss": total_value_loss / total_batches,
        "final_lr": scheduler.get_last_lr()[0],
    }


def get_temperature(iteration: int, total_iterations: int, initial_temp: float = 1.0, final_temp: float = 0.3) -> float:
    """Anneal temperature from initial to final over training."""
    progress = iteration / total_iterations
    return initial_temp - (initial_temp - final_temp) * progress


def main():
    parser = argparse.ArgumentParser(description="Enhanced self-play training v3")
    parser.add_argument("--iterations", type=int, default=30, help="Number of training iterations")
    parser.add_argument("--games-per-iteration", type=int, default=1000, help="Games to generate per iteration")
    parser.add_argument("--epochs-per-iteration", type=int, default=15, help="Training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--num-blocks", type=int, default=3, help="Number of residual blocks")
    parser.add_argument("--output-dir", type=str, default="checkpoints/selfplay-v3", help="Output directory")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--replay-buffer-size", type=int, default=5, help="Number of iterations to keep in replay buffer")
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

    # Create model
    model = EnhancedMLP(
        input_size=85,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        dropout=0.1
    )

    start_iteration = 1
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            # Try to load, but handle architecture mismatch
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as e:
                logger.warning(f"Could not load checkpoint (architecture mismatch): {e}")
                logger.info("Starting with fresh weights")
        if "iteration" in checkpoint:
            start_iteration = checkpoint["iteration"] + 1

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Replay buffer for diverse training
    replay_buffer = []

    # Training history
    history = []
    best_minimax_score = 0

    # Initial evaluation
    logger.info("Initial evaluation...")
    win_rate_random = evaluate_vs_random(model, device, num_games=30)
    win_rate_minimax = evaluate_vs_minimax(model, device, depth=2, num_games=10)
    logger.info(f"Initial: {win_rate_random:.1%} vs random, {win_rate_minimax:.1%} vs minimax(d=2)")

    # Training loop
    for iteration in range(start_iteration, args.iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}/{args.iterations}")
        logger.info(f"{'='*60}")

        # Calculate temperature for this iteration
        temperature = get_temperature(iteration, args.iterations)
        logger.info(f"Temperature: {temperature:.2f}")

        # Calculate learning rate decay
        lr = args.lr * (0.5 ** (iteration // 10))  # Halve every 10 iterations
        logger.info(f"Learning rate: {lr:.6f}")

        # Generate self-play games
        logger.info(f"Generating {args.games_per_iteration} self-play games...")
        start_time = time.time()

        neural_model = NeuralModel(model, device)
        worker = SelfPlayWorker(
            model=neural_model,
            temperature=temperature,
            temperature_threshold=15,  # Use temperature for first 15 moves
            add_noise=True,
        )
        games = worker.play_games(args.games_per_iteration)

        gen_time = time.time() - start_time
        logger.info(f"Generated {len(games)} games in {gen_time:.1f}s")

        # Game statistics
        total_positions = sum(len(g.positions) for g in games)
        p1_wins = sum(1 for g in games if g.metadata.get("result") == "player1_win")
        p2_wins = sum(1 for g in games if g.metadata.get("result") == "player2_win")
        draws = sum(1 for g in games if g.metadata.get("result") == "draw")
        logger.info(f"Positions: {total_positions}, P1: {p1_wins}, P2: {p2_wins}, Draws: {draws}")

        # Save games
        games_path = output_dir / f"games_iter_{iteration:03d}.jsonl"
        save_games_jsonl(games, str(games_path))

        # Update replay buffer
        replay_buffer.append(games)
        if len(replay_buffer) > args.replay_buffer_size:
            replay_buffer.pop(0)

        # Combine all games in replay buffer for training
        all_games = [g for games_list in replay_buffer for g in games_list]
        logger.info(f"Training on {len(all_games)} games from replay buffer")

        # Train on accumulated data
        logger.info(f"Training for {args.epochs_per_iteration} epochs...")
        start_time = time.time()

        metrics = train_iteration(
            model=model,
            games=all_games,
            device=device,
            epochs=args.epochs_per_iteration,
            batch_size=args.batch_size,
            lr=lr,
        )

        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.1f}s")
        logger.info(f"Loss: {metrics['avg_loss']:.4f} (policy: {metrics['avg_policy_loss']:.4f}, value: {metrics['avg_value_loss']:.4f})")

        # Evaluate
        logger.info("Evaluating...")
        win_rate_random = evaluate_vs_random(model, device, num_games=30)
        win_rate_minimax = evaluate_vs_minimax(model, device, depth=2, num_games=10)
        logger.info(f"Win rate: {win_rate_random:.1%} vs random, {win_rate_minimax:.1%} vs minimax(d=2)")

        # Save checkpoint
        checkpoint_path = output_dir / f"model_iter_{iteration:03d}.pt"
        torch.save({
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "win_rate_vs_random": win_rate_random,
            "win_rate_vs_minimax": win_rate_minimax,
            "temperature": temperature,
        }, checkpoint_path)

        # Save best model
        if win_rate_minimax > best_minimax_score:
            best_minimax_score = win_rate_minimax
            best_path = output_dir / "best_model.pt"
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "win_rate_vs_random": win_rate_random,
                "win_rate_vs_minimax": win_rate_minimax,
            }, best_path)
            logger.info(f"New best model! ({win_rate_minimax:.1%} vs minimax)")

        # Record history
        history.append({
            "iteration": iteration,
            "games": len(games),
            "positions": total_positions,
            "loss": metrics["avg_loss"],
            "policy_loss": metrics["avg_policy_loss"],
            "value_loss": metrics["avg_value_loss"],
            "win_rate_vs_random": win_rate_random,
            "win_rate_vs_minimax": win_rate_minimax,
            "temperature": temperature,
            "lr": lr,
            "gen_time": gen_time,
            "train_time": train_time,
        })

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": {
            "hidden_size": args.hidden_size,
            "num_blocks": args.num_blocks,
            "iterations": args.iterations,
        },
    }, final_path)
    logger.info(f"\nFinal model saved: {final_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    for h in history:
        logger.info(f"Iter {h['iteration']:2d}: loss={h['loss']:.4f}, random={h['win_rate_vs_random']:.1%}, minimax={h['win_rate_vs_minimax']:.1%}")

    logger.info(f"\nBest minimax score: {best_minimax_score:.1%}")


if __name__ == "__main__":
    main()
