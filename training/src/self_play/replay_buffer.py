"""
Replay Buffer for Self-Play Training

Fixed-size buffer with uniform sampling for experience replay.
"""

import random
import threading
from collections import deque
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..data import (
    COLUMNS,
    EncodingType,
    GameRecord,
    Position,
    encode_position,
    flip_board_horizontal,
    flip_move_horizontal,
    get_legal_moves,
)


class ReplayBuffer:
    """
    Fixed-size replay buffer with uniform sampling.

    Thread-safe for concurrent add/sample operations.
    """

    def __init__(self, max_size: int = 100000):
        """
        Initialize the replay buffer.

        Args:
            max_size: Maximum number of positions to store.
        """
        self.max_size = max_size
        self.buffer: deque[Position] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, position: Position) -> None:
        """
        Add a single position to the buffer.

        Args:
            position: Position to add.
        """
        with self._lock:
            self.buffer.append(position)

    def add_game(self, game: GameRecord) -> None:
        """
        Add all positions from a game to the buffer.

        Args:
            game: Game record containing positions.
        """
        with self._lock:
            for position in game.positions:
                self.buffer.append(position)

    def add_games(self, games: list[GameRecord]) -> None:
        """
        Add all positions from multiple games to the buffer.

        Args:
            games: List of game records.
        """
        with self._lock:
            for game in games:
                for position in game.positions:
                    self.buffer.append(position)

    def sample(self, batch_size: int) -> list[Position]:
        """
        Sample random positions from the buffer.

        Args:
            batch_size: Number of positions to sample.

        Returns:
            List of sampled positions.
        """
        with self._lock:
            if len(self.buffer) == 0:
                return []
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def sample_batch(
        self,
        batch_size: int,
        encoding: EncodingType = "flat-binary",
        augment: bool = True,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Sample a batch of positions as PyTorch tensors.

        Args:
            batch_size: Number of positions to sample.
            encoding: Position encoding type.
            augment: Whether to apply random horizontal flipping.
            device: Device to place tensors on.

        Returns:
            Dictionary with 'board', 'move', 'value', 'legal_mask', and 'policy' tensors.
        """
        positions = self.sample(batch_size)
        if not positions:
            return {}

        boards = []
        moves = []
        values = []
        legal_masks = []
        policies = []

        for pos in positions:
            # Random augmentation
            flip = augment and random.random() < 0.5

            if flip:
                board = flip_board_horizontal(pos.board)
                move = flip_move_horizontal(pos.move_played)
                policy = (
                    list(reversed(pos.policy_target))
                    if pos.policy_target
                    else None
                )
            else:
                board = pos.board
                move = pos.move_played
                policy = pos.policy_target

            # Encode board
            encoded = encode_position(board, pos.to_move, encoding, pos.moves_to_end)
            boards.append(encoded)
            moves.append(move)
            values.append(pos.result)

            # Legal moves mask
            legal = get_legal_moves(board)
            mask = np.zeros(COLUMNS, dtype=np.float32)
            for col in legal:
                mask[col] = 1.0
            legal_masks.append(mask)

            # Policy target
            if policy:
                policies.append(np.array(policy, dtype=np.float32))
            else:
                # Create one-hot from move played
                policy_arr = np.zeros(COLUMNS, dtype=np.float32)
                policy_arr[move] = 1.0
                policies.append(policy_arr)

        return {
            "board": torch.tensor(np.stack(boards), device=device),
            "move": torch.tensor(moves, dtype=torch.long, device=device),
            "value": torch.tensor(values, dtype=torch.float32, device=device),
            "legal_mask": torch.tensor(np.stack(legal_masks), device=device),
            "policy": torch.tensor(np.stack(policies), device=device),
        }

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all positions from the buffer."""
        with self._lock:
            self.buffer.clear()

    def get_statistics(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary of statistics.
        """
        with self._lock:
            if not self.buffer:
                return {
                    "size": 0,
                    "max_size": self.max_size,
                    "fill_ratio": 0.0,
                }

            results = {"win": 0, "loss": 0, "draw": 0}
            for pos in self.buffer:
                if pos.result > 0:
                    results["win"] += 1
                elif pos.result < 0:
                    results["loss"] += 1
                else:
                    results["draw"] += 1

            return {
                "size": len(self.buffer),
                "max_size": self.max_size,
                "fill_ratio": len(self.buffer) / self.max_size,
                "result_distribution": results,
            }


class ReplayBufferDataset(IterableDataset):
    """
    PyTorch IterableDataset wrapper for ReplayBuffer.

    Supports infinite iteration with random sampling.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        encoding: EncodingType = "flat-binary",
        augment: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            buffer: ReplayBuffer to sample from.
            encoding: Position encoding type.
            augment: Whether to apply data augmentation.
        """
        self.buffer = buffer
        self.encoding = encoding
        self.augment = augment

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate indefinitely, yielding random samples.

        Yields:
            Dictionary with position data.
        """
        while True:
            if len(self.buffer) == 0:
                continue

            positions = self.buffer.sample(1)
            if not positions:
                continue

            pos = positions[0]

            # Random augmentation
            flip = self.augment and random.random() < 0.5

            if flip:
                board = flip_board_horizontal(pos.board)
                move = flip_move_horizontal(pos.move_played)
                policy = (
                    list(reversed(pos.policy_target))
                    if pos.policy_target
                    else None
                )
            else:
                board = pos.board
                move = pos.move_played
                policy = pos.policy_target

            # Encode board
            encoded = encode_position(board, pos.to_move, self.encoding, pos.moves_to_end)

            # Legal moves mask
            legal = get_legal_moves(board)
            mask = np.zeros(COLUMNS, dtype=np.float32)
            for col in legal:
                mask[col] = 1.0

            # Policy target
            if policy:
                policy_arr = np.array(policy, dtype=np.float32)
            else:
                policy_arr = np.zeros(COLUMNS, dtype=np.float32)
                policy_arr[move] = 1.0

            yield {
                "board": torch.from_numpy(encoded),
                "move": torch.tensor(move, dtype=torch.long),
                "value": torch.tensor(pos.result, dtype=torch.float32),
                "legal_mask": torch.from_numpy(mask),
                "policy": torch.from_numpy(policy_arr),
            }
