#!/usr/bin/env python3
"""
Fast Self-Play Training with Parallel Game Generation

Optimizations:
- Multiprocessing for parallel game generation
- Batched neural network inference
- Larger training batches for GPU utilization
- Curriculum learning with minimax opponents
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import (
    ConnectFourDataset,
    save_games_jsonl,
    encode_flat_binary,
    COLUMNS,
)
from src.data.game import check_winner, ConnectFourGame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class FastMLP(nn.Module):
    """Optimized MLP for fast inference."""

    def __init__(self, input_size=85, hidden_sizes=None, dropout=0.1):
        super().__init__()
        hidden_sizes = hidden_sizes or [512, 256, 128]

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
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        shared = self.shared(x)
        return self.policy_head(shared), self.value_head(shared)


# Global model weights for worker processes
_global_weights = None


def init_worker(weights_dict):
    """Initialize worker with model weights."""
    global _global_weights
    _global_weights = weights_dict


def play_single_game(game_idx, temperature=1.0, temp_threshold=15, use_minimax_opponent=False, minimax_depth=2):
    """Play a single game (runs in worker process)."""
    global _global_weights

    # Recreate model in worker
    model = FastMLP(input_size=85, hidden_sizes=[512, 256, 128], dropout=0.0)
    model.load_state_dict(_global_weights)
    model.eval()

    board = [[None] * 7 for _ in range(6)]
    positions = []
    to_move = 1
    move_count = 0

    while True:
        # Check terminal
        winner = check_winner(board)
        if winner is not None:
            result = 1.0 if winner == 1 else -1.0
            break
        if all(board[0][col] is not None for col in range(7)):
            result = 0.0
            break

        # Get valid moves
        valid_moves = [col for col in range(7) if board[0][col] is None]
        if not valid_moves:
            result = 0.0
            break

        # Decide which agent moves
        if use_minimax_opponent and to_move == 2:
            # Use minimax for player 2
            move = minimax_move(board, to_move, minimax_depth)
            policy = [0.0] * 7
            policy[move] = 1.0
        else:
            # Neural network move
            encoded = encode_flat_binary(board, to_move)
            with torch.no_grad():
                tensor = torch.from_numpy(encoded).unsqueeze(0)
                policy_logits, value = model(tensor)
                probs = torch.softmax(policy_logits, dim=-1).numpy()[0]

            # Mask invalid moves
            for col in range(7):
                if board[0][col] is not None:
                    probs[col] = 0
            total = probs.sum()
            if total > 0:
                probs = probs / total
            else:
                probs = np.array([1.0/len(valid_moves) if col in valid_moves else 0 for col in range(7)])

            # Add Dirichlet noise
            noise = np.random.dirichlet([0.3] * len(valid_moves))
            noise_full = np.zeros(7)
            for i, col in enumerate(valid_moves):
                noise_full[col] = noise[i]
            probs = 0.75 * probs + 0.25 * noise_full

            # Temperature sampling
            temp = temperature if move_count < temp_threshold else 0.0
            if temp > 0:
                probs = probs ** (1.0 / temp)
                probs = probs / probs.sum()
                move = np.random.choice(7, p=probs)
            else:
                move = int(np.argmax(probs))

            policy = probs.tolist()

        # Record position
        positions.append({
            "board": [row[:] for row in board],
            "to_move": to_move,
            "move": move,
            "policy": policy,
        })

        # Make move
        for row in range(5, -1, -1):
            if board[row][move] is None:
                board[row][move] = to_move
                break

        to_move = 3 - to_move
        move_count += 1

    # Annotate with results
    for i, pos in enumerate(positions):
        pos["result"] = result if pos["to_move"] == 1 else -result
        pos["moves_to_end"] = len(positions) - i

    return {
        "positions": positions,
        "result": result,
        "num_moves": len(positions),
    }


def minimax_move(board, player, depth):
    """Simple minimax for opponent."""
    valid_moves = [col for col in range(7) if board[0][col] is None]
    if not valid_moves:
        return 0

    best_move = valid_moves[0]
    best_score = float('-inf')

    for move in valid_moves:
        # Make move
        new_board = [row[:] for row in board]
        for row in range(5, -1, -1):
            if new_board[row][move] is None:
                new_board[row][move] = player
                break

        score = -minimax_search(new_board, 3 - player, depth - 1, float('-inf'), float('inf'))
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def minimax_search(board, player, depth, alpha, beta):
    """Minimax with alpha-beta pruning."""
    winner = check_winner(board)
    if winner == player:
        return 1000 + depth
    elif winner is not None:
        return -1000 - depth

    valid_moves = [col for col in range(7) if board[0][col] is None]
    if not valid_moves or depth <= 0:
        return evaluate_board(board, player)

    best_score = float('-inf')
    for move in valid_moves:
        new_board = [row[:] for row in board]
        for row in range(5, -1, -1):
            if new_board[row][move] is None:
                new_board[row][move] = player
                break

        score = -minimax_search(new_board, 3 - player, depth - 1, -beta, -alpha)
        best_score = max(best_score, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best_score


def evaluate_board(board, player):
    """Simple position evaluation."""
    score = 0
    # Center column preference
    for row in range(6):
        if board[row][3] == player:
            score += 3
        elif board[row][3] is not None:
            score -= 3
    return score


def generate_games_parallel(model, num_games, num_workers, temperature, use_minimax=False, minimax_depth=2):
    """Generate games in parallel using multiprocessing."""
    weights = model.state_dict()

    # Convert to CPU for sharing
    weights_cpu = {k: v.cpu() for k, v in weights.items()}

    games = []
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(weights_cpu,)) as executor:
        futures = [
            executor.submit(play_single_game, i, temperature, 15, use_minimax, minimax_depth)
            for i in range(num_games)
        ]

        for future in as_completed(futures):
            try:
                game = future.result()
                games.append(game)
            except Exception as e:
                logger.warning(f"Game failed: {e}")

    return games


def games_to_dataset_format(games):
    """Convert games to format expected by ConnectFourDataset."""
    from src.data import GameRecord, Position

    records = []
    for game in games:
        positions = []
        for pos in game["positions"]:
            positions.append(Position(
                board=pos["board"],
                to_move=pos["to_move"],
                move_played=pos["move"],
                result=pos["result"],
                moves_to_end=pos["moves_to_end"],
                policy_target=pos["policy"],
                value_target=pos["result"],  # Use actual outcome
            ))
        records.append(GameRecord(
            game_id=str(len(records)),
            positions=positions,
            metadata={"result": game["result"]},
        ))
    return records


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0

    for batch in loader:
        boards = batch["board"].to(device)
        moves = batch["move"].to(device)
        values = batch["value"].to(device)

        policy_logits, value_pred = model(boards)
        policy_loss = nn.functional.cross_entropy(policy_logits, moves)
        value_loss = nn.functional.mse_loss(value_pred.squeeze(), values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
    }


def evaluate_model(model, device, num_games=20):
    """Evaluate against random and minimax."""
    model.eval()

    # vs Random
    random_wins = 0
    for i in range(num_games):
        result = play_eval_game(model, device, opponent="random", neural_first=(i % 2 == 0))
        if result > 0:
            random_wins += 1

    # vs Minimax depth 2
    minimax_wins = 0
    for i in range(num_games):
        result = play_eval_game(model, device, opponent="minimax", minimax_depth=2, neural_first=(i % 2 == 0))
        if result > 0:
            minimax_wins += 1

    return {
        "vs_random": random_wins / num_games,
        "vs_minimax": minimax_wins / num_games,
    }


def play_eval_game(model, device, opponent="random", minimax_depth=2, neural_first=True):
    """Play evaluation game. Returns 1 if neural wins, -1 if loses, 0 draw."""
    board = [[None] * 7 for _ in range(6)]
    to_move = 1
    neural_player = 1 if neural_first else 2

    while True:
        winner = check_winner(board)
        if winner is not None:
            return 1 if winner == neural_player else -1
        if all(board[0][col] is not None for col in range(7)):
            return 0

        valid_moves = [col for col in range(7) if board[0][col] is None]

        if to_move == neural_player:
            # Neural move
            encoded = encode_flat_binary(board, to_move)
            with torch.no_grad():
                tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
                policy_logits, _ = model(tensor)
                probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]

            for col in range(7):
                if board[0][col] is not None:
                    probs[col] = 0
            move = int(np.argmax(probs))
        else:
            if opponent == "random":
                move = np.random.choice(valid_moves)
            else:
                move = minimax_move(board, to_move, minimax_depth)

        for row in range(5, -1, -1):
            if board[row][move] is None:
                board[row][move] = to_move
                break

        to_move = 3 - to_move


def main():
    parser = argparse.ArgumentParser(description="Fast parallel self-play training")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games-per-iteration", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--output-dir", type=str, default="checkpoints/selfplay-fast")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum with minimax opponents")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}, Workers: {args.workers}")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = FastMLP(input_size=85, hidden_sizes=[512, 256, 128], dropout=0.1)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)

    # Replay buffer
    replay_buffer = []
    best_minimax_score = 0

    # Initial eval
    eval_results = evaluate_model(model, device, num_games=20)
    logger.info(f"Initial: {eval_results['vs_random']:.1%} vs random, {eval_results['vs_minimax']:.1%} vs minimax")

    for iteration in range(1, args.iterations + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"ITERATION {iteration}/{args.iterations}")
        logger.info(f"{'='*50}")

        # Temperature annealing
        temperature = max(0.3, 1.0 - 0.7 * (iteration / args.iterations))

        # Curriculum: add minimax opponents after iteration 10
        use_minimax = args.curriculum and iteration > 10
        minimax_ratio = min(0.5, (iteration - 10) / 20) if use_minimax else 0

        # Generate games in parallel
        logger.info(f"Generating {args.games_per_iteration} games (temp={temperature:.2f}, minimax={minimax_ratio:.0%})...")
        start = time.time()

        # Split between self-play and vs minimax
        num_minimax = int(args.games_per_iteration * minimax_ratio)
        num_selfplay = args.games_per_iteration - num_minimax

        games = []
        if num_selfplay > 0:
            games.extend(generate_games_parallel(model, num_selfplay, args.workers, temperature, use_minimax=False))
        if num_minimax > 0:
            games.extend(generate_games_parallel(model, num_minimax, args.workers, temperature, use_minimax=True, minimax_depth=2))

        gen_time = time.time() - start
        positions = sum(g["num_moves"] for g in games)
        logger.info(f"Generated {len(games)} games, {positions} positions in {gen_time:.1f}s ({positions/gen_time:.0f} pos/s)")

        # Convert to dataset format
        records = games_to_dataset_format(games)
        replay_buffer.extend(records)
        if len(replay_buffer) > 10000:  # Keep last 10k games
            replay_buffer = replay_buffer[-10000:]

        # Train
        dataset = ConnectFourDataset(games=replay_buffer, encoding="flat-binary", augment=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        logger.info(f"Training on {len(dataset)} positions...")
        start = time.time()
        for epoch in range(args.epochs):
            metrics = train_epoch(model, loader, optimizer, device)
        train_time = time.time() - start
        scheduler.step()

        logger.info(f"Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f}) in {train_time:.1f}s")

        # Evaluate
        eval_results = evaluate_model(model, device, num_games=20)
        logger.info(f"Eval: {eval_results['vs_random']:.1%} vs random, {eval_results['vs_minimax']:.1%} vs minimax")

        # Save checkpoint
        torch.save({
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "eval": eval_results,
        }, output_dir / f"model_iter_{iteration:03d}.pt")

        # Best model
        if eval_results['vs_minimax'] > best_minimax_score:
            best_minimax_score = eval_results['vs_minimax']
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"New best! {best_minimax_score:.1%} vs minimax")

    # Final save
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    logger.info(f"\nTraining complete. Best vs minimax: {best_minimax_score:.1%}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
