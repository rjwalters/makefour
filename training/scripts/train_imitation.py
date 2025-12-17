#!/usr/bin/env python3
"""
Imitation Learning: Train Neural Network on Minimax Expert Games

This script generates expert-level games using minimax players and trains
a neural network to imitate their play. This is much more effective than
pure self-play because the neural network learns from consistently strong moves.

Usage:
    python scripts/train_imitation.py --games 10000 --depth 5 --epochs 50
"""

import argparse
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import COLUMNS, ROWS, Board, Player, encode_flat_binary
from src.evaluation.agents import (
    MinimaxAgent,
    get_legal_moves,
    check_winner,
    apply_move,
    minimax_search,
    evaluate_position,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single position from a game."""
    board: Board
    player: Player
    best_move: int  # Minimax best move
    result: float   # Game result from this player's perspective


@dataclass
class Game:
    """A complete game with positions."""
    positions: list[Position]
    winner: Player | None


# Global minimax depth for workers
_worker_depth: int = 5


def init_worker(depth: int):
    """Initialize worker with minimax depth."""
    global _worker_depth
    _worker_depth = depth


def get_minimax_move(board: Board, player: Player, depth: int) -> int:
    """Get the best move according to minimax."""
    legal_moves = get_legal_moves(board)
    if len(legal_moves) == 1:
        return legal_moves[0]

    _, best_move = minimax_search(
        board=board,
        depth=depth,
        alpha=float("-inf"),
        beta=float("inf"),
        maximizing=True,
        player=player,
        current_player=player,
    )
    return best_move if best_move is not None else legal_moves[0]


def get_opponent_move(board: Board, player: Player, opponent_depth: int) -> int:
    """Get a move for the opponent at specified depth (0 = random)."""
    legal_moves = get_legal_moves(board)
    if len(legal_moves) == 1:
        return legal_moves[0]

    if opponent_depth == 0:
        # Random opponent
        return random.choice(legal_moves)
    else:
        # Minimax opponent
        return get_minimax_move(board, player, opponent_depth)


def play_expert_game(game_id: int, depth: int = 5, add_noise: bool = True) -> Game:
    """
    Play a game between two minimax players (self-play).

    Args:
        game_id: Game identifier (for seeding)
        depth: Minimax search depth
        add_noise: Add occasional random moves for diversity

    Returns:
        Game with all positions and results
    """
    random.seed(game_id)

    board: Board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
    current_player: Player = 1
    positions: list[Position] = []

    while True:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            break

        # Get minimax best move
        best_move = get_minimax_move(board, current_player, depth)

        # Occasionally add noise for training diversity (10% of moves)
        if add_noise and random.random() < 0.1 and len(legal_moves) > 1:
            # Pick a random move, but not completely random - prefer good moves
            # Use shallower search to rank moves
            move_scores = []
            for move in legal_moves:
                new_board = apply_move(board, move, current_player)
                score = evaluate_position(new_board, current_player)
                move_scores.append((move, score))

            # Sort by score and pick from top half
            move_scores.sort(key=lambda x: x[1], reverse=True)
            top_moves = [m for m, s in move_scores[:max(2, len(move_scores) // 2)]]
            actual_move = random.choice(top_moves)
        else:
            actual_move = best_move

        # Store position with the BEST move (what we want to learn)
        positions.append(Position(
            board=[row[:] for row in board],  # Copy board
            player=current_player,
            best_move=best_move,
            result=0.0,  # Will be set after game ends
        ))

        # Make move
        board = apply_move(board, actual_move, current_player)

        # Check for winner
        winner = check_winner(board)
        if winner is not None:
            break

        # Switch player
        current_player = 2 if current_player == 1 else 1

    # Determine winner
    final_winner = check_winner(board)
    if final_winner == "draw":
        game_winner = None
    else:
        game_winner = final_winner

    # Set results for all positions
    for pos in positions:
        if game_winner is None:
            pos.result = 0.0  # Draw
        elif game_winner == pos.player:
            pos.result = 1.0  # Win
        else:
            pos.result = -1.0  # Loss

    return Game(positions=positions, winner=game_winner)


def play_mixed_opponent_game(
    game_id: int,
    expert_depth: int = 5,
    opponent_depth: int = 2,
) -> Game:
    """
    Play a game where expert plays against a weaker opponent.

    Only expert positions are recorded for training - we want to learn
    how to beat players of various skill levels.

    Args:
        game_id: Game identifier (for seeding)
        expert_depth: Minimax search depth for expert
        opponent_depth: Depth for opponent (0 = random)

    Returns:
        Game with expert positions and results
    """
    random.seed(game_id)

    board: Board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
    current_player: Player = 1
    positions: list[Position] = []

    # Expert plays as player 1 in half the games
    expert_player: Player = 1 if game_id % 2 == 0 else 2

    while True:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            break

        if current_player == expert_player:
            # Expert move - record position
            best_move = get_minimax_move(board, current_player, expert_depth)

            positions.append(Position(
                board=[row[:] for row in board],
                player=current_player,
                best_move=best_move,
                result=0.0,
            ))

            actual_move = best_move
        else:
            # Opponent move - don't record
            actual_move = get_opponent_move(board, current_player, opponent_depth)

        # Make move
        board = apply_move(board, actual_move, current_player)

        # Check for winner
        winner = check_winner(board)
        if winner is not None:
            break

        # Switch player
        current_player = 2 if current_player == 1 else 1

    # Determine winner
    final_winner = check_winner(board)
    if final_winner == "draw":
        game_winner = None
    else:
        game_winner = final_winner

    # Set results for expert positions
    for pos in positions:
        if game_winner is None:
            pos.result = 0.0  # Draw
        elif game_winner == pos.player:
            pos.result = 1.0  # Win
        else:
            pos.result = -1.0  # Loss

    return Game(positions=positions, winner=game_winner)


def play_single_game(args: tuple) -> Game:
    """Wrapper for parallel execution."""
    game_id, depth, add_noise = args
    return play_expert_game(game_id, depth, add_noise)


def play_single_mixed_game(args: tuple) -> Game:
    """Wrapper for parallel mixed game execution."""
    game_id, expert_depth, opponent_depth = args
    return play_mixed_opponent_game(game_id, expert_depth, opponent_depth)


def generate_expert_games(
    num_games: int,
    depth: int = 5,
    num_workers: int = 8,
    add_noise: bool = True,
) -> list[Game]:
    """Generate games between minimax players in parallel."""
    logger.info(f"Generating {num_games} expert games (depth={depth}, workers={num_workers})")

    args_list = [(i, depth, add_noise) for i in range(num_games)]
    games = []

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(play_single_game, args) for args in args_list]

        for future in tqdm(as_completed(futures), total=num_games, desc="Games"):
            games.append(future.result())

    elapsed = time.time() - start_time
    total_positions = sum(len(g.positions) for g in games)

    logger.info(f"Generated {num_games} games, {total_positions} positions in {elapsed:.1f}s")
    logger.info(f"Speed: {total_positions / elapsed:.0f} positions/sec")

    # Stats
    wins_p1 = sum(1 for g in games if g.winner == 1)
    wins_p2 = sum(1 for g in games if g.winner == 2)
    draws = sum(1 for g in games if g.winner is None)
    logger.info(f"Results: P1 wins {wins_p1}, P2 wins {wins_p2}, Draws {draws}")

    return games


def generate_mixed_games(
    num_games: int,
    expert_depth: int = 5,
    opponent_depths: list[int] = None,
    num_workers: int = 8,
) -> list[Game]:
    """
    Generate games where expert plays against opponents of varying skill.

    Games are distributed evenly across opponent depths.

    Args:
        num_games: Total number of games to generate
        expert_depth: Minimax depth for expert player
        opponent_depths: List of depths for opponents (0=random, 1-4=minimax)
        num_workers: Parallel workers
    """
    if opponent_depths is None:
        opponent_depths = [0, 1, 2, 3, 4]  # Random + depths 1-4

    logger.info(f"Generating {num_games} mixed games (expert_depth={expert_depth})")
    logger.info(f"Opponent depths: {opponent_depths}")

    # Distribute games across opponent depths
    games_per_depth = num_games // len(opponent_depths)
    args_list = []
    game_id = 0

    for opp_depth in opponent_depths:
        for _ in range(games_per_depth):
            args_list.append((game_id, expert_depth, opp_depth))
            game_id += 1

    # Add remaining games
    remaining = num_games - len(args_list)
    for i in range(remaining):
        opp_depth = opponent_depths[i % len(opponent_depths)]
        args_list.append((game_id, expert_depth, opp_depth))
        game_id += 1

    games = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(play_single_mixed_game, args) for args in args_list]

        for future in tqdm(as_completed(futures), total=len(args_list), desc="Mixed Games"):
            games.append(future.result())

    elapsed = time.time() - start_time
    total_positions = sum(len(g.positions) for g in games)

    logger.info(f"Generated {num_games} games, {total_positions} positions in {elapsed:.1f}s")
    logger.info(f"Speed: {total_positions / elapsed:.0f} positions/sec")

    # Stats by opponent depth
    for opp_depth in opponent_depths:
        depth_games = [g for i, g in enumerate(games) if args_list[i][2] == opp_depth]
        wins = sum(1 for g in depth_games if g.winner is not None)
        logger.info(f"  vs depth-{opp_depth}: {len(depth_games)} games, expert won {wins}")

    return games


def generate_hybrid_games(
    num_games: int,
    expert_depth: int = 5,
    expert_ratio: float = 0.5,
    opponent_depths: list[int] = None,
    num_workers: int = 8,
    add_noise: bool = True,
) -> list[Game]:
    """
    Generate a mix of expert self-play and mixed opponent games.

    This gives the model exposure to both optimal play (for strength)
    and weaker opponents (for learning to punish mistakes).

    Args:
        num_games: Total number of games
        expert_depth: Minimax depth for expert
        expert_ratio: Fraction of games that are expert vs expert
        opponent_depths: Depths for mixed opponent games
        num_workers: Parallel workers
        add_noise: Add noise to expert games for diversity
    """
    if opponent_depths is None:
        opponent_depths = [0, 1, 2, 3]  # Don't include depth-4 (too similar to expert)

    num_expert_games = int(num_games * expert_ratio)
    num_mixed_games = num_games - num_expert_games

    logger.info(f"Generating hybrid dataset: {num_expert_games} expert + {num_mixed_games} mixed games")

    # Build combined args list for better load balancing
    args_list = []
    game_id = 0

    # Expert self-play games
    for _ in range(num_expert_games):
        args_list.append(('expert', game_id, expert_depth, add_noise))
        game_id += 1

    # Mixed opponent games - distribute across depths
    games_per_depth = num_mixed_games // len(opponent_depths)
    for opp_depth in opponent_depths:
        for _ in range(games_per_depth):
            args_list.append(('mixed', game_id, expert_depth, opp_depth))
            game_id += 1

    # Remaining mixed games
    remaining = num_mixed_games - (games_per_depth * len(opponent_depths))
    for i in range(remaining):
        opp_depth = opponent_depths[i % len(opponent_depths)]
        args_list.append(('mixed', game_id, expert_depth, opp_depth))
        game_id += 1

    # Shuffle for better load balancing (mix fast/slow games)
    random.shuffle(args_list)

    games = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(play_hybrid_game, args) for args in args_list]

        for future in tqdm(as_completed(futures), total=len(args_list), desc="Hybrid Games"):
            games.append(future.result())

    elapsed = time.time() - start_time
    total_positions = sum(len(g.positions) for g in games)

    logger.info(f"Generated {num_games} games, {total_positions} positions in {elapsed:.1f}s")
    logger.info(f"Speed: {total_positions / elapsed:.0f} positions/sec")

    return games


def play_hybrid_game(args: tuple) -> Game:
    """Wrapper for hybrid game generation."""
    game_type = args[0]
    if game_type == 'expert':
        _, game_id, depth, add_noise = args
        return play_expert_game(game_id, depth, add_noise)
    else:
        _, game_id, expert_depth, opp_depth = args
        return play_mixed_opponent_game(game_id, expert_depth, opp_depth)


class ImitationMLP(nn.Module):
    """MLP for imitation learning with policy and value heads."""

    def __init__(self, input_size: int = 85, hidden_sizes: list[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 256, 128]

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        # Policy head - predict best move
        self.policy_head = nn.Linear(prev_size, COLUMNS)

        # Value head - predict game outcome
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        shared = self.shared(x)
        policy = self.policy_head(shared)
        value = self.value_head(shared)
        return policy, value


def prepare_training_data(games: list[Game], device: torch.device):
    """Convert games to training tensors."""
    inputs = []
    policy_targets = []
    value_targets = []

    for game in games:
        for pos in game.positions:
            # Encode position
            encoded = encode_flat_binary(pos.board, pos.player)
            inputs.append(encoded)

            # Policy target: one-hot for best move
            policy = np.zeros(COLUMNS, dtype=np.float32)
            policy[pos.best_move] = 1.0
            policy_targets.append(policy)

            # Value target
            value_targets.append(pos.result)

    X = torch.tensor(np.array(inputs), dtype=torch.float32, device=device)
    y_policy = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
    y_value = torch.tensor(np.array(value_targets), dtype=torch.float32, device=device).unsqueeze(1)

    return X, y_policy, y_value


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y_policy: torch.Tensor,
    y_value: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 0.001,
    device: torch.device = None,
):
    """Train the model on expert data."""
    dataset = TensorDataset(X, y_policy, y_value)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for batch_x, batch_policy, batch_value in dataloader:
            optimizer.zero_grad()

            policy_pred, value_pred = model(batch_x)

            # Policy loss (cross-entropy with soft targets)
            policy_loss = policy_criterion(policy_pred, batch_policy)

            # Value loss
            value_loss = value_criterion(value_pred, batch_value)

            # Combined loss (policy is more important for move prediction)
            loss = policy_loss + 0.5 * value_loss

            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        scheduler.step()

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}"
            )

    return model


def evaluate_vs_minimax(model: nn.Module, depth: int = 4, num_games: int = 100, device: torch.device = None):
    """Evaluate the trained model against minimax."""
    model.eval()

    wins = 0
    losses = 0
    draws = 0

    for game_i in range(num_games):
        board: Board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
        current_player: Player = 1

        # Neural plays as player 1 in half the games
        neural_player: Player = 1 if game_i % 2 == 0 else 2

        while True:
            legal_moves = get_legal_moves(board)
            if not legal_moves:
                break

            if current_player == neural_player:
                # Neural network move
                with torch.no_grad():
                    encoded = encode_flat_binary(board, current_player)
                    x = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
                    policy, _ = model(x)

                    # Mask illegal moves
                    policy = policy.cpu().numpy()[0]
                    masked = np.full(COLUMNS, float("-inf"))
                    for m in legal_moves:
                        masked[m] = policy[m]

                    move = int(np.argmax(masked))
            else:
                # Minimax move
                move = get_minimax_move(board, current_player, depth)

            board = apply_move(board, move, current_player)

            winner = check_winner(board)
            if winner is not None:
                if winner == "draw":
                    draws += 1
                elif winner == neural_player:
                    wins += 1
                else:
                    losses += 1
                break

            current_player = 2 if current_player == 1 else 1

    win_rate = wins / num_games * 100
    logger.info(f"vs Minimax(d={depth}): {wins}W {losses}L {draws}D ({win_rate:.1f}% win rate)")

    return wins, losses, draws


def export_onnx(model: nn.Module, output_path: str, device: torch.device):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 85, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=13,
    )

    logger.info(f"Exported ONNX model to {output_path}")


def main():
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(description="Train neural network on minimax expert games")
    parser.add_argument("--games", type=int, default=10000, help="Number of games to generate")
    parser.add_argument("--depth", type=int, default=5, help="Minimax search depth for experts")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size (larger = better GPU utilization)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--workers", type=int, default=cpu_count, help=f"Parallel workers (default: {cpu_count} CPUs)")
    parser.add_argument("--output", type=str, default="models/imitation-v1.onnx", help="Output model path")
    parser.add_argument("--no-noise", action="store_true", help="Don't add noise to games")
    parser.add_argument("--mixed", action="store_true", help="Train against mixed opponents (random, depth 1-4)")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid: mix of expert self-play and mixed opponents")
    parser.add_argument("--expert-ratio", type=float, default=0.5, help="Ratio of expert games in hybrid mode")
    parser.add_argument("--opponent-depths", type=str, default="0,1,2,3,4",
                       help="Comma-separated opponent depths for mixed mode (0=random)")
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Using {args.workers} parallel workers for game generation")

    # Generate games
    if args.hybrid:
        # Hybrid: mix of expert self-play and mixed opponents
        opponent_depths = [int(d) for d in args.opponent_depths.split(",")]
        # Remove depth-4+ from mixed (too similar to expert)
        opponent_depths = [d for d in opponent_depths if d < 4]
        games = generate_hybrid_games(
            num_games=args.games,
            expert_depth=args.depth,
            expert_ratio=args.expert_ratio,
            opponent_depths=opponent_depths,
            num_workers=args.workers,
            add_noise=not args.no_noise,
        )
    elif args.mixed:
        # Mixed opponent training only
        opponent_depths = [int(d) for d in args.opponent_depths.split(",")]
        games = generate_mixed_games(
            num_games=args.games,
            expert_depth=args.depth,
            opponent_depths=opponent_depths,
            num_workers=args.workers,
        )
    else:
        # Expert self-play (original mode)
        games = generate_expert_games(
            num_games=args.games,
            depth=args.depth,
            num_workers=args.workers,
            add_noise=not args.no_noise,
        )

    # Prepare training data
    logger.info("Preparing training data...")
    X, y_policy, y_value = prepare_training_data(games, device)
    logger.info(f"Training data: {X.shape[0]} positions")

    # Create model
    model = ImitationMLP(input_size=85, hidden_sizes=[256, 256, 128]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Train
    logger.info("Training model...")
    model = train_model(
        model, X, y_policy, y_value,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Evaluate
    logger.info("\nEvaluating model...")
    evaluate_vs_minimax(model, depth=2, num_games=100, device=device)
    evaluate_vs_minimax(model, depth=3, num_games=100, device=device)
    evaluate_vs_minimax(model, depth=4, num_games=50, device=device)

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, str(output_path), device)

    logger.info(f"\nTraining complete! Model saved to {output_path}")


if __name__ == "__main__":
    main()
