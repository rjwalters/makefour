#!/usr/bin/env python3
"""
Oracle-Guided ResNet Training for Connect Four

Deeper ResNet architecture for stronger play (~1650 ELO target).
Uses residual blocks and curriculum learning with increasing oracle depth.

Usage:
    python scripts/train_oracle_resnet.py --games 10000 --epochs 120 --curriculum
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import COLUMNS, ROWS, Board, Player, encode_onehot
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
    """Get soft labels from minimax oracle."""
    legal_moves = get_legal_moves(board)
    if not legal_moves:
        return np.zeros(COLUMNS, dtype=np.float32)

    move_scores = np.full(COLUMNS, float("-inf"), dtype=np.float32)

    for col in legal_moves:
        new_board = apply_move(board, col, player)
        winner = check_winner(new_board)

        if winner == player:
            move_scores[col] = 1000.0
        elif winner == 3 - player:
            move_scores[col] = -1000.0
        else:
            opponent = 3 - player
            score, _ = minimax_search(
                board=new_board,
                depth=depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=False,
                player=player,
                current_player=opponent,
            )
            move_scores[col] = score  # Minimax already returns from player's perspective

    valid_mask = move_scores > float("-inf")
    if not np.any(valid_mask):
        return np.zeros(COLUMNS, dtype=np.float32)

    valid_scores = move_scores[valid_mask]
    max_score = np.max(valid_scores)
    scaled = (valid_scores - max_score) / max(temperature, 0.01)
    exp_scores = np.exp(np.clip(scaled, -20, 0))

    probs = np.zeros(COLUMNS, dtype=np.float32)
    probs[valid_mask] = exp_scores / np.sum(exp_scores)

    return probs


# -----------------------------------------------------------------------------
# Data Generation
# -----------------------------------------------------------------------------

@dataclass
class OraclePosition:
    """A position with oracle soft labels."""
    board: Board
    player: Player
    soft_labels: np.ndarray


def generate_oracle_game(depth: int, temperature: float = 0.3) -> list[OraclePosition]:
    """Generate one game with oracle evaluation at each position."""
    positions = []
    board = [[None] * COLUMNS for _ in range(ROWS)]
    current_player = 1

    while True:
        winner = check_winner(board)
        if winner is not None:
            break

        legal_moves = get_legal_moves(board)
        if not legal_moves:
            break

        soft_labels = get_oracle_soft_labels(board, current_player, depth, temperature)

        positions.append(OraclePosition(
            board=[row[:] for row in board],
            player=current_player,
            soft_labels=soft_labels.copy(),
        ))

        if random.random() < 0.1:
            move = random.choice(legal_moves)
        else:
            probs = soft_labels.copy()
            probs[probs < 0] = 0
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
                move = np.random.choice(COLUMNS, p=probs)
            else:
                move = random.choice(legal_moves)

        board = apply_move(board, move, current_player)
        current_player = 3 - current_player

    return positions


def generate_games_parallel(
    num_games: int,
    depth: int,
    workers: int = 8,
    temperature: float = 0.3,
) -> list[OraclePosition]:
    """Generate games in parallel."""
    all_positions = []
    completed = 0
    log_interval = max(1, num_games // 20)  # Log every 5%

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(generate_oracle_game, depth, temperature)
            for _ in range(num_games)
        ]

        for future in as_completed(futures):
            try:
                positions = future.result(timeout=120)
                all_positions.extend(positions)
                completed += 1
                if completed % log_interval == 0 or completed == num_games:
                    pct = 100.0 * completed / num_games
                    logger.info(f"Generating (d={depth}): {completed}/{num_games} ({pct:.0f}%)")
            except Exception as e:
                logger.warning(f"Game generation failed: {e}")
                completed += 1

    return all_positions


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class OracleDataset(Dataset):
    """Dataset of positions with oracle soft labels."""

    def __init__(self, positions: list[OraclePosition], augment: bool = True):
        self.positions = positions
        self.augment = augment

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        pos = self.positions[idx]
        board = pos.board
        player = pos.player

        flip = self.augment and random.random() < 0.5
        if flip:
            board = [row[::-1] for row in board]

        encoded = encode_onehot(board, player)
        x = torch.tensor(encoded, dtype=torch.float32)

        soft_labels = pos.soft_labels
        if flip:
            soft_labels = soft_labels[::-1].copy()
        y = torch.tensor(soft_labels, dtype=torch.float32)

        return x, y


# -----------------------------------------------------------------------------
# ResNet Model
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class OracleResNet(nn.Module):
    """ResNet for oracle-guided training with one-hot input."""

    def __init__(
        self,
        num_blocks: int = 6,
        channels: int = 128,
        policy_hidden: int = 256,
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Flatten size: channels * 6 * 7
        flat_size = channels * 6 * 7

        # Policy head with more capacity
        self.policy_head = nn.Sequential(
            nn.Linear(flat_size, policy_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(policy_hidden, policy_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(policy_hidden // 2, COLUMNS),
        )

    def forward(self, x):
        # x shape: (batch, 126) - reshape to (batch, 3, 6, 7)
        if x.dim() == 2:
            x = x.view(-1, 3, 6, 7)

        features = self.input_conv(x)
        features = self.res_blocks(features)
        features = features.view(features.size(0), -1)
        policy = self.policy_head(features)
        return policy


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def kl_divergence_loss(pred_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """KL divergence loss for soft label training."""
    pred_probs = F.softmax(pred_logits, dim=-1)
    target_probs = target_probs + 1e-8
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)

    log_pred = torch.log(pred_probs + 1e-8)
    log_target = torch.log(target_probs)

    kl = torch.sum(target_probs * (log_target - log_pred), dim=-1)
    return kl.mean()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    total_steps: int = None,
    steps_taken: list = None,
) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top3 = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = kl_divergence_loss(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if steps_taken is not None and total_steps is not None:
            if steps_taken[0] < total_steps:
                scheduler.step()
                steps_taken[0] += 1
        else:
            try:
                scheduler.step()
            except ValueError:
                pass

        pred_top1 = logits.argmax(dim=-1)
        target_top1 = y.argmax(dim=-1)
        total_correct_top1 += (pred_top1 == target_top1).sum().item()

        _, pred_top3 = logits.topk(3, dim=-1)
        for i in range(len(y)):
            if target_top1[i] in pred_top3[i]:
                total_correct_top3 += 1

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    top1_acc = total_correct_top1 / total_samples
    top3_acc = total_correct_top3 / total_samples

    return avg_loss, top1_acc, top3_acc


def evaluate_against_minimax(
    model: nn.Module,
    depth: int,
    num_games: int,
    device: torch.device,
) -> Tuple[int, int, int]:
    """Evaluate model against minimax opponent."""
    model.eval()
    wins, losses, draws = 0, 0, 0

    for game_idx in range(num_games):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        neural_player = 1 if game_idx % 2 == 0 else 2
        current_player = 1

        while True:
            winner = check_winner(board)
            if winner is not None:
                if winner == neural_player:
                    wins += 1
                else:
                    losses += 1
                break

            legal_moves = get_legal_moves(board)
            if not legal_moves:
                draws += 1
                break

            if current_player == neural_player:
                with torch.no_grad():
                    encoded = encode_onehot(board, current_player)
                    x = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = model(x)

                    probs = logits.cpu().numpy()[0]
                    for col in range(COLUMNS):
                        if col not in legal_moves:
                            probs[col] = float("-inf")
                    move = int(np.argmax(probs))
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
                    move = random.choice(legal_moves)

            board = apply_move(board, move, current_player)
            current_player = 3 - current_player

    return wins, losses, draws


def export_onnx(model: nn.Module, output_path: str, device: torch.device):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 126, device=device)

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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(description="Oracle-guided ResNet training")
    parser.add_argument("--games", type=int, default=10000, help="Games per depth level")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Max learning rate")
    parser.add_argument("--workers", type=int, default=cpu_count, help="Parallel workers")
    parser.add_argument("--output", type=str, default="models/resnet-oracle-v1.onnx", help="Output path")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--start-depth", type=int, default=3, help="Starting oracle depth")
    parser.add_argument("--end-depth", type=int, default=7, help="Final oracle depth")
    parser.add_argument("--num-blocks", type=int, default=6, help="Number of residual blocks")
    parser.add_argument("--channels", type=int, default=128, help="Channel width")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Create model
    model = OracleResNet(
        num_blocks=args.num_blocks,
        channels=args.channels,
        policy_hidden=256,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: OracleResNet with {total_params:,} parameters")
    logger.info(f"Architecture: {args.num_blocks} blocks, {args.channels} channels")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Generate initial data
    all_positions = []
    current_depth = args.start_depth if args.curriculum else args.end_depth

    logger.info(f"Generating initial data at depth {current_depth}...")
    positions = generate_games_parallel(
        args.games, current_depth, args.workers, temperature=0.5
    )
    all_positions.extend(positions)
    logger.info(f"Generated {len(positions)} positions")

    # Create dataset and dataloader
    num_workers = min(4, cpu_count)
    dataset = OracleDataset(all_positions)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu"),
        persistent_workers=(num_workers > 0),
    )
    logger.info(f"DataLoader: batch_size={args.batch_size}, num_workers={num_workers}")

    # Estimate total steps for scheduler
    estimated_growth = 1.8 if args.curriculum else 1.0
    estimated_positions = int(len(all_positions) * estimated_growth)
    estimated_batches_per_epoch = max(1, estimated_positions // args.batch_size)
    total_steps = args.epochs * estimated_batches_per_epoch

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )
    steps_taken = [0]

    best_win_rate = 0.0

    for epoch in range(1, args.epochs + 1):
        # Curriculum: increase depth
        if args.curriculum:
            progress = epoch / args.epochs
            new_depth = args.start_depth + int(progress * (args.end_depth - args.start_depth))

            if new_depth > current_depth:
                current_depth = new_depth
                logger.info(f"Curriculum: increasing to depth {current_depth}")

                new_positions = generate_games_parallel(
                    args.games // 5, current_depth, args.workers, temperature=0.3
                )
                all_positions.extend(new_positions)
                logger.info(f"Added {len(new_positions)} positions, total: {len(all_positions)}")

                dataset = OracleDataset(all_positions)
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=(device.type != "cpu"),
                    persistent_workers=(num_workers > 0),
                )

        # Train
        loss, top1, top3 = train_epoch(
            model, dataloader, optimizer, scheduler, device,
            total_steps=total_steps, steps_taken=steps_taken
        )

        # Evaluate periodically
        if epoch % 20 == 0 or epoch == 1:
            w3, l3, d3 = evaluate_against_minimax(model, 3, 30, device)
            w4, l4, d4 = evaluate_against_minimax(model, 4, 30, device)

            wr3 = w3 / max(w3 + l3 + d3, 1) * 100
            wr4 = w4 / max(w4 + l4 + d4, 1) * 100

            logger.info(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | Top1: {top1*100:.1f}% | Top3: {top3*100:.1f}% | "
                f"vs d3: {wr3:.0f}% | vs d4: {wr4:.0f}%"
            )

            if wr4 > best_win_rate:
                best_win_rate = wr4
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "win_rate_d4": wr4,
                }, "models/best_resnet_model.pt")
                logger.info(f"New best! Win rate vs d4: {wr4:.0f}%")
        else:
            logger.info(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Top1: {top1*100:.1f}% | Top3: {top3*100:.1f}%")

    # Final evaluation
    logger.info("Final evaluation...")
    for depth in [3, 4, 5, 6]:
        w, l, d = evaluate_against_minimax(model, depth, 100, device)
        wr = w / max(w + l + d, 1) * 100
        logger.info(f"vs Minimax(d={depth}): {w}W/{l}L/{d}D ({wr:.1f}%)")

    # Export
    export_onnx(model, args.output, device)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
