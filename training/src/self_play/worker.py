"""
Self-Play Worker

Generates games through self-play with configurable exploration strategies.
"""

import random
import uuid
from typing import Protocol, runtime_checkable

import numpy as np

from ..data import COLUMNS, ConnectFourGame, GameRecord, Position


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for neural network models used in self-play."""

    def predict(self, board: list[list[int | None]], to_move: int) -> tuple[np.ndarray, float]:
        """
        Predict policy and value for a position.

        Args:
            board: 6x7 board state
            to_move: Current player (1 or 2)

        Returns:
            Tuple of (policy, value) where policy is a 7-element array
            of move probabilities and value is the position evaluation.
        """
        ...


class RandomModel:
    """A random policy model for testing and baseline comparisons."""

    def predict(
        self, board: list[list[int | None]], to_move: int
    ) -> tuple[np.ndarray, float]:
        """Returns uniform random policy over legal moves."""
        # Find legal moves
        legal_moves = [col for col in range(COLUMNS) if board[0][col] is None]

        policy = np.zeros(COLUMNS, dtype=np.float32)
        if legal_moves:
            prob = 1.0 / len(legal_moves)
            for col in legal_moves:
                policy[col] = prob

        # Random value
        value = 0.0

        return policy, value


class SelfPlayWorker:
    """
    Generates games through self-play.

    Supports exploration via temperature-based sampling and Dirichlet noise.
    """

    def __init__(
        self,
        model: ModelProtocol | None = None,
        temperature: float = 1.0,
        temperature_threshold: int = 15,
        add_noise: bool = True,
        noise_alpha: float = 0.3,
        noise_epsilon: float = 0.25,
    ):
        """
        Initialize the self-play worker.

        Args:
            model: Neural network model for policy prediction. If None, uses random policy.
            temperature: Temperature for move sampling. Higher = more exploration.
            temperature_threshold: After this many moves, temperature drops to 0.
            add_noise: Whether to add Dirichlet noise to root policy.
            noise_alpha: Dirichlet distribution alpha parameter.
            noise_epsilon: Weight of noise in final policy (1-eps)*policy + eps*noise.
        """
        self.model = model if model is not None else RandomModel()
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.add_noise = add_noise
        self.noise_alpha = noise_alpha
        self.noise_epsilon = noise_epsilon

    def play_game(self) -> GameRecord:
        """
        Play a complete game through self-play.

        Returns:
            GameRecord containing all positions and the game outcome.
        """
        game = ConnectFourGame()
        positions: list[Position] = []

        while not game.is_terminal():
            # Get model policy
            policy, value = self.model.predict(game.board, game.current_player)

            # Mask illegal moves
            legal_moves = game.get_legal_moves()
            policy = self._mask_illegal_moves(policy, legal_moves)

            # Add exploration noise at root (Dirichlet noise for AlphaZero-style exploration)
            if self.add_noise:
                policy = self._add_dirichlet_noise(policy, legal_moves)

            # Determine temperature based on move count
            move_count = len(game.move_history)
            temp = self.temperature if move_count < self.temperature_threshold else 0.0

            # Sample move with temperature
            move = self._sample_move(policy, temp)

            # Record position BEFORE making move
            positions.append(
                Position(
                    board=[row[:] for row in game.board],
                    to_move=game.current_player,
                    move_played=move,
                    result=0.0,  # Will be updated after game ends
                    moves_to_end=0,  # Will be updated after game ends
                    policy_target=policy.tolist(),
                    value_target=value,
                )
            )

            game.make_move(move)

        # Annotate positions with game outcome
        result = game.get_result()  # From player 1's perspective
        total_moves = len(positions)

        for i, pos in enumerate(positions):
            # Value from perspective of player to move at that position
            pos.result = result if pos.to_move == 1 else -result
            pos.moves_to_end = total_moves - i

        # Create game record
        game_id = str(uuid.uuid4())
        metadata = {
            "temperature": self.temperature,
            "temperature_threshold": self.temperature_threshold,
            "noise_enabled": self.add_noise,
            "noise_alpha": self.noise_alpha,
            "noise_epsilon": self.noise_epsilon,
            "total_moves": total_moves,
            "result": "player1_win" if result > 0 else "player2_win" if result < 0 else "draw",
        }

        return GameRecord(
            game_id=game_id,
            positions=positions,
            metadata=metadata,
        )

    def _mask_illegal_moves(
        self, policy: np.ndarray, legal_moves: list[int]
    ) -> np.ndarray:
        """
        Mask illegal moves and renormalize policy.

        Args:
            policy: Raw policy from model.
            legal_moves: List of legal column indices.

        Returns:
            Normalized policy with illegal moves zeroed out.
        """
        masked = np.zeros(COLUMNS, dtype=np.float32)
        for col in legal_moves:
            masked[col] = policy[col]

        # Renormalize
        total = masked.sum()
        if total > 0:
            masked /= total
        elif legal_moves:
            # If all legal moves have zero probability, use uniform
            prob = 1.0 / len(legal_moves)
            for col in legal_moves:
                masked[col] = prob

        return masked

    def _add_dirichlet_noise(
        self, policy: np.ndarray, legal_moves: list[int]
    ) -> np.ndarray:
        """
        Add Dirichlet noise for exploration (AlphaZero-style).

        Args:
            policy: Current policy distribution.
            legal_moves: List of legal column indices.

        Returns:
            Policy with added noise.
        """
        if not legal_moves:
            return policy

        # Generate Dirichlet noise for legal moves only
        noise = np.random.dirichlet([self.noise_alpha] * len(legal_moves))

        # Mix noise with policy
        result = policy.copy()
        for i, col in enumerate(legal_moves):
            result[col] = (1 - self.noise_epsilon) * policy[col] + self.noise_epsilon * noise[i]

        # Renormalize
        total = result.sum()
        if total > 0:
            result /= total

        return result

    def _sample_move(self, policy: np.ndarray, temperature: float) -> int:
        """
        Sample a move from the policy distribution with temperature.

        Args:
            policy: Probability distribution over moves.
            temperature: Sampling temperature. 0 = argmax, higher = more random.

        Returns:
            Selected column index.
        """
        if temperature == 0:
            return int(np.argmax(policy))

        # Apply temperature
        scaled = np.power(policy, 1.0 / temperature)
        total = scaled.sum()

        if total == 0:
            # All probabilities became zero after scaling, use uniform over non-zero
            non_zero = np.where(policy > 0)[0]
            if len(non_zero) > 0:
                return int(np.random.choice(non_zero))
            return int(np.random.choice(COLUMNS))

        scaled /= total
        return int(np.random.choice(COLUMNS, p=scaled))

    def play_games(self, num_games: int) -> list[GameRecord]:
        """
        Play multiple games.

        Args:
            num_games: Number of games to play.

        Returns:
            List of GameRecord objects.
        """
        return [self.play_game() for _ in range(num_games)]
