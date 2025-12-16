"""
PyTorch Dataset for Connect Four Training

Provides dataset classes for loading and preprocessing Connect Four game data.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .encoding import (
    COLUMNS,
    Board,
    EncodingType,
    Player,
    board_from_moves,
    encode_position,
    flip_board_horizontal,
    flip_move_horizontal,
)
from .game import ConnectFourGame, get_legal_moves


@dataclass
class Position:
    """A single game position for training."""

    board: Board
    to_move: Player
    move_played: int  # Column 0-6
    result: float  # 1.0=player1 wins, -1.0=player2 wins, 0.0=draw
    moves_to_end: int  # Depth to terminal
    policy_target: list[float] | None = None  # Optional: from perfect solver
    value_target: float | None = None  # Optional: position evaluation


@dataclass
class GameRecord:
    """A complete game record for training."""

    game_id: str
    positions: list[Position]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts to JSON-serializable dictionary."""
        return {
            "game_id": self.game_id,
            "positions": [
                {
                    "board": pos.board,
                    "to_move": pos.to_move,
                    "move_played": pos.move_played,
                    "result": pos.result,
                    "moves_to_end": pos.moves_to_end,
                    "policy_target": pos.policy_target,
                    "value_target": pos.value_target,
                }
                for pos in self.positions
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameRecord":
        """Creates from JSON dictionary."""
        positions = [
            Position(
                board=pos["board"],
                to_move=pos["to_move"],
                move_played=pos["move_played"],
                result=pos["result"],
                moves_to_end=pos["moves_to_end"],
                policy_target=pos.get("policy_target"),
                value_target=pos.get("value_target"),
            )
            for pos in data["positions"]
        ]
        return cls(
            game_id=data["game_id"],
            positions=positions,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_moves(
        cls,
        game_id: str,
        moves: list[int],
        metadata: dict | None = None,
    ) -> "GameRecord":
        """
        Creates a game record from a sequence of moves.

        Args:
            game_id: Unique identifier for the game.
            moves: List of column indices (0-6) representing each move.
            metadata: Optional metadata dictionary.

        Returns:
            GameRecord with all positions from the game.
        """
        game = ConnectFourGame.from_moves(moves)
        result = game.get_result()  # From player 1's perspective

        positions = []
        board: Board = [[None for _ in range(COLUMNS)] for _ in range(6)]
        current_player: Player = 1

        for i, move in enumerate(moves):
            # Record the position BEFORE the move is made
            moves_to_end = len(moves) - i

            # Value from current player's perspective
            value = result if current_player == 1 else -result

            positions.append(
                Position(
                    board=[row[:] for row in board],  # Deep copy
                    to_move=current_player,
                    move_played=move,
                    result=value,
                    moves_to_end=moves_to_end,
                )
            )

            # Make the move
            for row in range(5, -1, -1):
                if board[row][move] is None:
                    board[row][move] = current_player
                    break

            # Switch player
            current_player = 2 if current_player == 1 else 1

        return cls(
            game_id=game_id,
            positions=positions,
            metadata=metadata or {},
        )


class ConnectFourDataset(Dataset):
    """
    PyTorch Dataset for Connect Four positions.

    Loads game records and provides individual positions for training.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        games: list[GameRecord] | None = None,
        encoding: EncodingType = "flat-binary",
        augment: bool = True,
        balance_results: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to JSONL file(s) containing game records.
            games: Pre-loaded list of GameRecord objects.
            encoding: Position encoding type.
            augment: Whether to apply data augmentation (horizontal flip).
            balance_results: Whether to balance by game result.
        """
        self.encoding = encoding
        self.augment = augment
        self.balance_results = balance_results

        # Load positions
        self.positions: list[tuple[Position, bool]] = []  # (position, is_flipped)

        if games is not None:
            self._load_games(games)
        elif data_path is not None:
            self._load_from_path(data_path)

    def _load_games(self, games: list[GameRecord]) -> None:
        """Load positions from game records."""
        for game in games:
            for pos in game.positions:
                self.positions.append((pos, False))
                if self.augment:
                    self.positions.append((pos, True))

    def _load_from_path(self, data_path: str | Path) -> None:
        """Load game records from JSONL file(s)."""
        data_path = Path(data_path)

        if data_path.is_file():
            self._load_jsonl_file(data_path)
        elif data_path.is_dir():
            for file_path in data_path.glob("*.jsonl"):
                self._load_jsonl_file(file_path)
        else:
            raise ValueError(f"Invalid data path: {data_path}")

    def _load_jsonl_file(self, file_path: Path) -> None:
        """Load game records from a single JSONL file."""
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    game = GameRecord.from_dict(data)
                    for pos in game.positions:
                        self.positions.append((pos, False))
                        if self.augment:
                            self.positions.append((pos, True))

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        pos, is_flipped = self.positions[idx]

        # Get board and move (potentially flipped)
        if is_flipped:
            board = flip_board_horizontal(pos.board)
            move = flip_move_horizontal(pos.move_played)
        else:
            board = pos.board
            move = pos.move_played

        # Encode the position
        encoded = encode_position(board, pos.to_move, self.encoding, pos.moves_to_end)

        # Get legal moves mask
        legal_moves = get_legal_moves(board)
        legal_mask = np.zeros(COLUMNS, dtype=np.float32)
        for col in legal_moves:
            legal_mask[col] = 1.0

        # Value target (from current player's perspective)
        value = pos.value_target if pos.value_target is not None else pos.result

        return {
            "board": torch.from_numpy(encoded),
            "move": torch.tensor(move, dtype=torch.long),
            "value": torch.tensor(value, dtype=torch.float32),
            "legal_mask": torch.from_numpy(legal_mask),
            "to_move": pos.to_move,
            "moves_to_end": pos.moves_to_end,
        }

    def get_statistics(self) -> dict:
        """Returns dataset statistics."""
        results = {"win": 0, "loss": 0, "draw": 0}

        for pos, _ in self.positions:
            if pos.result > 0:
                results["win"] += 1
            elif pos.result < 0:
                results["loss"] += 1
            else:
                results["draw"] += 1

        return {
            "total_positions": len(self.positions),
            "unique_positions": len(self.positions) // 2 if self.augment else len(self.positions),
            "result_distribution": results,
            "encoding": self.encoding,
            "augmented": self.augment,
        }


def save_games_jsonl(games: list[GameRecord], file_path: str | Path) -> None:
    """Save game records to a JSONL file."""
    with open(file_path, "w") as f:
        for game in games:
            f.write(json.dumps(game.to_dict()) + "\n")


def load_games_jsonl(file_path: str | Path) -> list[GameRecord]:
    """Load game records from a JSONL file."""
    games = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                games.append(GameRecord.from_dict(data))
    return games


def create_train_val_test_split(
    games: list[GameRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[GameRecord], list[GameRecord], list[GameRecord]]:
    """
    Splits games into train/val/test sets.

    Args:
        games: List of game records to split.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_games, val_games, test_games).
    """
    random.seed(seed)
    games = games[:]  # Copy to avoid modifying original
    random.shuffle(games)

    n = len(games)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        games[:train_end],
        games[train_end:val_end],
        games[val_end:],
    )
