#!/usr/bin/env python3
"""
Oracle-Guided Teacher Training for Connect Four Neural Networks

This script implements teacher-student training where a minimax oracle
provides soft labels during training. The neural network learns not just
the best move, but a distribution over move quality.

Key features:
- Soft labels from minimax evaluation (move quality distribution)
- Curriculum learning (oracle depth increases over epochs)
- Online oracle queries for positions the model struggles with
- Temperature scaling for controlling label smoothness

Usage:
    python scripts/train_oracle.py --games 5000 --epochs 100 --curriculum
"""

import argparse
import logging
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import COLUMNS, ROWS, Board, Player, encode_flat_binary
from src.evaluation.agents import (
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


# -----------------------------------------------------------------------------
# Oracle Functions
# -----------------------------------------------------------------------------

def get_oracle_soft_labels(
    board: Board,
    player: Player,
    depth: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Get soft labels from minimax oracle.

    Instead of just returning the best move, we evaluate ALL legal moves
    and return a probability distribution based on their quality.

    Args:
        board: Current board state
        player: Player to move
        depth: Minimax search depth
        temperature: Controls distribution sharpness (lower = sharper)

    Returns:
        Probability distribution over columns (7 values)
    """
    legal_moves = get_legal_moves(board)

    if not legal_moves:
        return np.zeros(COLUMNS, dtype=np.float32)

    if len(legal_moves) == 1:
        # Only one legal move - deterministic
        probs = np.zeros(COLUMNS, dtype=np.float32)
        probs[legal_moves[0]] = 1.0
        return probs

    # Evaluate each legal move with minimax
    move_scores = []
    for move in legal_moves:
        new_board = apply_move(board, move, player)

        # Check for immediate win
        winner = check_winner(new_board)
        if winner == player:
            # Winning move - return it deterministically
            probs = np.zeros(COLUMNS, dtype=np.float32)
            probs[move] = 1.0
            return probs

        # Evaluate position after move
        if depth > 0:
            opponent = 2 if player == 1 else 1
            score, _ = minimax_search(
                board=new_board,
                depth=depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=False,  # Opponent's turn
                player=player,
                current_player=opponent,
            )
            # Negate because we evaluated from opponent's perspective
            move_scores.append((move, score))
        else:
            # Depth 0: use static evaluation
            score = evaluate_position(new_board, player)
            move_scores.append((move, score))

    # Convert scores to probabilities using softmax with temperature
    scores = np.array([s for _, s in move_scores], dtype=np.float32)

    # Normalize scores to prevent overflow
    scores = scores - scores.max()

    # Apply temperature and softmax
    exp_scores = np.exp(scores / temperature)
    probs_legal = exp_scores / exp_scores.sum()

    # Map back to full probability distribution
    probs = np.zeros(COLUMNS, dtype=np.float32)
    for (move, _), p in zip(move_scores, probs_legal):
        probs[move] = p

    return probs


def generate_oracle_position(args: tuple) -> tuple:
    """Generate a single position with oracle soft labels."""
    game_id, depth, temperature, max_moves = args
    random.seed(game_id)

    board: Board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
    current_player: Player = 1

    positions = []
    move_count = 0

    while move_count < max_moves:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            break

        # Get oracle soft labels
        soft_labels = get_oracle_soft_labels(board, current_player, depth, temperature)

        # Store position
        positions.append({
            'board': [row[:] for row in board],
            'player': current_player,
            'soft_labels': soft_labels,
        })

        # Make a move (sample from oracle distribution for diversity)
        if random.random() < 0.9:
            # Usually pick best move
            move = int(np.argmax(soft_labels))
        else:
            # Sometimes sample from distribution
            move = np.random.choice(COLUMNS, p=soft_labels)

        if move not in legal_moves:
            move = legal_moves[0]

        board = apply_move(board, move, current_player)

        winner = check_winner(board)
        if winner is not None:
            break

        current_player = 2 if current_player == 1 else 1
        move_count += 1

    return positions


def generate_oracle_dataset(
    num_games: int,
    depth: int = 4,
    temperature: float = 1.0,
    max_moves: int = 42,
    num_workers: int = 8,
) -> list[dict]:
    """Generate dataset with oracle soft labels."""
    logger.info(f"Generating oracle dataset: {num_games} games, depth={depth}, temp={temperature}")

    args_list = [(i, depth, temperature, max_moves) for i in range(num_games)]
    all_positions = []

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_oracle_position, args) for args in args_list]

        for future in tqdm(as_completed(futures), total=num_games, desc="Oracle Games"):
            positions = future.result()
            all_positions.extend(positions)

    elapsed = time.time() - start_time
    logger.info(f"Generated {len(all_positions)} positions in {elapsed:.1f}s")
    logger.info(f"Speed: {len(all_positions) / elapsed:.0f} positions/sec")

    return all_positions


# -----------------------------------------------------------------------------
# Dataset with Augmentation
# -----------------------------------------------------------------------------

class OracleDataset(Dataset):
    """Dataset with oracle soft labels and augmentation."""

    def __init__(self, positions: list[dict], augment: bool = True):
        self.positions = positions
        self.augment = augment

    def __len__(self):
        return len(self.positions) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        # Handle augmentation indexing
        flip = False
        if self.augment:
            flip = idx >= len(self.positions)
            idx = idx % len(self.positions)

        pos = self.positions[idx]

        # Encode board
        board = pos['board']
        player = pos['player']

        if flip:
            # Horizontal flip
            board = [row[::-1] for row in board]

        encoded = encode_flat_binary(board, player)
        x = torch.tensor(encoded, dtype=torch.float32)

        # Soft labels (policy target)
        soft_labels = pos['soft_labels']
        if flip:
            soft_labels = soft_labels[::-1].copy()

        y = torch.tensor(soft_labels, dtype=torch.float32)

        return x, y


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class OracleNet(nn.Module):
    """Neural network for oracle-guided training."""

    def __init__(
        self,
        input_size: int = 85,
        hidden_sizes: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [512, 256, 128]

        layers = []
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            if i < len(hidden_sizes) - 1:  # No dropout on last hidden layer
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.backbone = nn.Sequential(*layers)

        # Policy head only (value can be derived from policy in many cases)
        self.policy_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.GELU(),
            nn.Linear(64, COLUMNS),
        )

    def forward(self, x):
        features = self.backbone(x)
        policy = self.policy_head(features)
        return policy


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def kl_divergence_loss(pred_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """
    KL divergence loss for soft label training.

    This is better than cross-entropy for soft labels because it properly
    handles the distribution matching objective.
    """
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
    # Add small epsilon to prevent log(0)
    target_probs = target_probs.clamp(min=1e-8)
    kl = F.kl_div(pred_log_probs, target_probs, reduction='batchmean')
    return kl


def cross_entropy_soft(pred_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets."""
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1).mean()
    return loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str = "kl",
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(batch_x)
                if loss_type == "kl":
                    loss = kl_divergence_loss(logits, batch_y)
                else:
                    loss = cross_entropy_soft(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            if loss_type == "kl":
                loss = kl_divergence_loss(logits, batch_y)
            else:
                loss = cross_entropy_soft(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model accuracy."""
    model.eval()

    total_top1 = 0
    total_top3 = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            preds = logits.argmax(dim=-1)
            targets = batch_y.argmax(dim=-1)

            total_top1 += (preds == targets).sum().item()

            # Top-3 accuracy
            _, top3_preds = logits.topk(3, dim=-1)
            for i in range(len(targets)):
                if targets[i] in top3_preds[i]:
                    total_top3 += 1

            total += len(targets)

    return total_top1 / total, total_top3 / total


def evaluate_vs_minimax(
    model: nn.Module,
    depth: int = 4,
    num_games: int = 100,
    device: torch.device = None,
):
    """Evaluate model against minimax."""
    model.eval()

    wins = 0
    losses = 0
    draws = 0

    for game_i in range(num_games):
        board: Board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
        current_player: Player = 1
        neural_player: Player = 1 if game_i % 2 == 0 else 2

        while True:
            legal_moves = get_legal_moves(board)
            if not legal_moves:
                break

            if current_player == neural_player:
                with torch.no_grad():
                    encoded = encode_flat_binary(board, current_player)
                    x = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = model(x)

                    # Mask illegal moves
                    probs = logits.cpu().numpy()[0]
                    masked = np.full(COLUMNS, float("-inf"))
                    for m in legal_moves:
                        masked[m] = probs[m]

                    move = int(np.argmax(masked))
            else:
                _, move = minimax_search(
                    board=board,
                    depth=depth,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    maximizing=True,
                    player=current_player,
                    current_player=current_player,
                )
                if move is None:
                    move = legal_moves[0]

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
    logger.info(f"vs Minimax(d={depth}): {wins}W {losses}L {draws}D ({win_rate:.1f}%)")

    return wins, losses, draws


# -----------------------------------------------------------------------------
# Main Training Loop with Curriculum
# -----------------------------------------------------------------------------

def train_with_curriculum(
    model: nn.Module,
    num_games: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_workers: int,
    start_depth: int = 2,
    end_depth: int = 5,
    temperature_schedule: str = "linear",
    output_dir: Path = None,
):
    """
    Train with curriculum learning using ACCUMULATED data.

    Key improvement: Instead of replacing data at each depth transition,
    we ACCUMULATE data from all depths. This prevents catastrophic forgetting.

    - Start with shallow oracle (easier to learn from)
    - Gradually add higher depth data while keeping old data
    - Use replay buffer to maintain knowledge from all depths
    """
    logger.info("=" * 60)
    logger.info("ORACLE-GUIDED CURRICULUM TRAINING (v2 - Accumulated)")
    logger.info("=" * 60)
    logger.info(f"Curriculum: depth {start_depth} → {end_depth}")
    logger.info(f"Epochs: {epochs}, Games: {num_games}, Batch: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=1,  # Epoch-level scheduling
        pct_start=0.1,
        anneal_strategy='cos',
    )

    # Mixed precision (for CUDA)
    scaler = None
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_accuracy = 0.0
    best_win_rate = 0.0

    # ACCUMULATED positions from all depths (replay buffer)
    all_positions: list[dict] = []
    positions_by_depth: dict[int, list[dict]] = {}
    prev_depth = -1

    for epoch in range(epochs):
        # Curriculum: compute current oracle depth
        progress = epoch / max(1, epochs - 1)
        current_depth = int(start_depth + progress * (end_depth - start_depth))
        current_depth = min(current_depth, end_depth)

        # Temperature: start high (soft), decrease over time
        if temperature_schedule == "linear":
            temperature = 2.0 - 1.5 * progress  # 2.0 → 0.5
        else:
            temperature = 1.0
        temperature = max(0.3, temperature)

        # Generate NEW data only when depth increases, but KEEP old data
        if current_depth != prev_depth:
            # Generate data at new depth
            games_for_depth = num_games // (end_depth - start_depth + 1)
            logger.info(f"\n[Epoch {epoch+1}] Adding depth={current_depth} data (temp={temperature:.2f})")

            new_positions = generate_oracle_dataset(
                num_games=games_for_depth,
                depth=current_depth,
                temperature=temperature,
                num_workers=num_workers,
            )

            # Store by depth and add to accumulated buffer
            positions_by_depth[current_depth] = new_positions
            all_positions.extend(new_positions)

            logger.info(f"Total accumulated positions: {len(all_positions)} from depths {list(positions_by_depth.keys())}")

            # Create dataset from ALL accumulated positions
            dataset = OracleDataset(all_positions, augment=True)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

        prev_depth = current_depth

        # Train epoch on ALL accumulated data
        loss = train_epoch(model, dataloader, optimizer, device, loss_type="kl", scaler=scaler)
        scheduler.step()

        # Evaluate
        top1_acc, top3_acc = evaluate_accuracy(model, dataloader, device)

        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"[Epoch {epoch+1}/{epochs}] Loss={loss:.4f}, "
            f"Top1={top1_acc*100:.1f}%, Top3={top3_acc*100:.1f}%, "
            f"LR={current_lr:.6f}, Depth={current_depth}"
        )

        # Periodic evaluation vs minimax (more frequent)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logger.info("\n--- Evaluation vs Minimax ---")
            wins3, _, _ = evaluate_vs_minimax(model, depth=3, num_games=50, device=device)
            wins4, _, _ = evaluate_vs_minimax(model, depth=4, num_games=50, device=device)

            # Save best model based on win rate vs depth-4
            current_win_rate = wins4 / 50
            if current_win_rate > best_win_rate and output_dir:
                best_win_rate = current_win_rate
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'win_rate_d4': current_win_rate,
                }, output_dir / "best_model.pt")
                logger.info(f"New best model saved (d4 win rate: {current_win_rate*100:.1f}%)")

        # Save best by accuracy
        if top1_acc > best_accuracy and output_dir:
            best_accuracy = top1_acc

    # Load best model for final evaluation
    if output_dir and (output_dir / "best_model.pt").exists():
        checkpoint = torch.load(output_dir / "best_model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

    return model


def save_dataset(positions: list[dict], path: Path):
    """Save dataset to disk for reuse."""
    with open(path, 'wb') as f:
        pickle.dump(positions, f)
    logger.info(f"Saved {len(positions)} positions to {path}")


def load_dataset(path: Path) -> list[dict]:
    """Load dataset from disk."""
    with open(path, 'rb') as f:
        positions = pickle.load(f)
    logger.info(f"Loaded {len(positions)} positions from {path}")
    return positions


def export_onnx(model: nn.Module, output_path: str, device: torch.device):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 85, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["policy"],
        dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}},
        opset_version=13,
    )
    logger.info(f"Exported ONNX model to {output_path}")


def main():
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(description="Oracle-guided teacher training")
    parser.add_argument("--games", type=int, default=5000, help="Games per epoch")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Max learning rate")
    parser.add_argument("--workers", type=int, default=cpu_count, help="Parallel workers")
    parser.add_argument("--output", type=str, default="models/oracle-v1.onnx", help="Output path")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--start-depth", type=int, default=2, help="Starting oracle depth")
    parser.add_argument("--end-depth", type=int, default=5, help="Final oracle depth")
    parser.add_argument("--hidden", type=str, default="512,256,128", help="Hidden layer sizes")
    parser.add_argument("--load-data", type=str, help="Load cached dataset")
    parser.add_argument("--save-data", type=str, help="Save dataset for reuse")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Output directory
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    hidden_sizes = [int(x) for x in args.hidden.split(",")]
    model = OracleNet(input_size=85, hidden_sizes=hidden_sizes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {hidden_sizes} ({total_params:,} parameters)")

    # Optionally compile with torch 2.0
    if hasattr(torch, 'compile') and device.type == "cuda":
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Train
    if args.curriculum:
        model = train_with_curriculum(
            model=model,
            num_games=args.games,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            num_workers=args.workers,
            start_depth=args.start_depth,
            end_depth=args.end_depth,
            output_dir=output_dir,
        )
    else:
        # Single depth training
        logger.info(f"Generating oracle dataset at depth {args.end_depth}...")

        if args.load_data:
            positions = load_dataset(Path(args.load_data))
        else:
            positions = generate_oracle_dataset(
                num_games=args.games,
                depth=args.end_depth,
                temperature=1.0,
                num_workers=args.workers,
            )
            if args.save_data:
                save_dataset(positions, Path(args.save_data))

        dataset = OracleDataset(positions, augment=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(args.epochs):
            loss = train_epoch(model, dataloader, optimizer, device)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                top1, top3 = evaluate_accuracy(model, dataloader, device)
                logger.info(f"[Epoch {epoch+1}] Loss={loss:.4f}, Top1={top1*100:.1f}%")

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    evaluate_vs_minimax(model, depth=2, num_games=100, device=device)
    evaluate_vs_minimax(model, depth=3, num_games=100, device=device)
    evaluate_vs_minimax(model, depth=4, num_games=100, device=device)
    evaluate_vs_minimax(model, depth=5, num_games=50, device=device)

    # Export
    # For ONNX export, we need base model (not compiled)
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    export_onnx(base_model, str(output_path), device)

    logger.info(f"\nTraining complete! Model saved to {output_path}")


if __name__ == "__main__":
    main()
