"""
Agent Implementations for Model Evaluation

Provides agent base class and wrappers for different types of players:
- RandomAgent: Uniform random move selection
- MinimaxAgent: Python minimax with alpha-beta pruning
- NeuralAgent: ONNX model inference
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..data import COLUMNS, ROWS, Board, Player, encode_onehot

# Type alias for game result
GameResult = Literal["player1_win", "player2_win", "draw"]


@dataclass
class AgentInfo:
    """Information about an agent."""

    name: str
    description: str
    expected_elo: int


class Agent(ABC):
    """
    Abstract base class for Connect Four agents.

    All agents must implement get_move() which returns a column index (0-6).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the agent's name."""
        pass

    @property
    @abstractmethod
    def info(self) -> AgentInfo:
        """Returns information about the agent."""
        pass

    @abstractmethod
    def get_move(self, board: Board, to_move: Player) -> int:
        """
        Select a move for the given position.

        Args:
            board: Current board state (6x7 grid)
            to_move: Current player (1 or 2)

        Returns:
            Column index (0-6) for the selected move
        """
        pass

    def reset(self) -> None:
        """Reset any internal state. Called before each game."""
        pass


def get_legal_moves(board: Board) -> list[int]:
    """Returns list of valid column indices where a piece can be dropped."""
    return [col for col in range(COLUMNS) if board[0][col] is None]


class RandomAgent(Agent):
    """Agent that plays uniformly random legal moves."""

    @property
    def name(self) -> str:
        return "random"

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name="Random",
            description="Plays uniformly random legal moves",
            expected_elo=0,
        )

    def get_move(self, board: Board, to_move: Player) -> int:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return random.choice(legal_moves)


# ============================================================================
# MINIMAX AGENT
# ============================================================================

# Evaluation weights matching TypeScript minimax-engine.ts
EVAL_WEIGHTS = {
    "WIN": 100000,
    "THREE_IN_ROW": 100,
    "TWO_IN_ROW": 10,
    "CENTER_CONTROL": 3,
}

WIN_LENGTH = 4


def evaluate_window(window: list[Player | None], player: Player) -> int:
    """Evaluates a window of 4 cells for scoring potential."""
    opponent: Player = 2 if player == 1 else 1
    player_count = sum(1 for c in window if c == player)
    opponent_count = sum(1 for c in window if c == opponent)
    empty_count = sum(1 for c in window if c is None)

    if opponent_count > 0 and player_count > 0:
        return 0

    if player_count == 4:
        return EVAL_WEIGHTS["WIN"]
    if player_count == 3 and empty_count == 1:
        return EVAL_WEIGHTS["THREE_IN_ROW"]
    if player_count == 2 and empty_count == 2:
        return EVAL_WEIGHTS["TWO_IN_ROW"]

    if opponent_count == 4:
        return -EVAL_WEIGHTS["WIN"]
    if opponent_count == 3 and empty_count == 1:
        return -EVAL_WEIGHTS["THREE_IN_ROW"]
    if opponent_count == 2 and empty_count == 2:
        return -EVAL_WEIGHTS["TWO_IN_ROW"]

    return 0


def evaluate_position(board: Board, player: Player) -> int:
    """Evaluates the board position from the perspective of the given player."""
    score = 0

    # Center column control
    center_col = COLUMNS // 2
    for row in range(ROWS):
        if board[row][center_col] == player:
            score += EVAL_WEIGHTS["CENTER_CONTROL"]
        elif board[row][center_col] is not None:
            score -= EVAL_WEIGHTS["CENTER_CONTROL"]

    # Horizontal windows
    for row in range(ROWS):
        for col in range(COLUMNS - WIN_LENGTH + 1):
            window = [board[row][col + i] for i in range(WIN_LENGTH)]
            score += evaluate_window(window, player)

    # Vertical windows
    for col in range(COLUMNS):
        for row in range(ROWS - WIN_LENGTH + 1):
            window = [board[row + i][col] for i in range(WIN_LENGTH)]
            score += evaluate_window(window, player)

    # Diagonal windows (down-right)
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(COLUMNS - WIN_LENGTH + 1):
            window = [board[row + i][col + i] for i in range(WIN_LENGTH)]
            score += evaluate_window(window, player)

    # Diagonal windows (down-left)
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(WIN_LENGTH - 1, COLUMNS):
            window = [board[row + i][col - i] for i in range(WIN_LENGTH)]
            score += evaluate_window(window, player)

    return score


def check_winner(board: Board) -> Player | Literal["draw"] | None:
    """Checks if there's a winner on the board."""
    # Check horizontal
    for row in range(ROWS):
        for col in range(COLUMNS - 3):
            cell = board[row][col]
            if cell is not None:
                if (
                    board[row][col + 1] == cell
                    and board[row][col + 2] == cell
                    and board[row][col + 3] == cell
                ):
                    return cell

    # Check vertical
    for row in range(ROWS - 3):
        for col in range(COLUMNS):
            cell = board[row][col]
            if cell is not None:
                if (
                    board[row + 1][col] == cell
                    and board[row + 2][col] == cell
                    and board[row + 3][col] == cell
                ):
                    return cell

    # Check diagonal (down-right)
    for row in range(ROWS - 3):
        for col in range(COLUMNS - 3):
            cell = board[row][col]
            if cell is not None:
                if (
                    board[row + 1][col + 1] == cell
                    and board[row + 2][col + 2] == cell
                    and board[row + 3][col + 3] == cell
                ):
                    return cell

    # Check diagonal (up-right)
    for row in range(3, ROWS):
        for col in range(COLUMNS - 3):
            cell = board[row][col]
            if cell is not None:
                if (
                    board[row - 1][col + 1] == cell
                    and board[row - 2][col + 2] == cell
                    and board[row - 3][col + 3] == cell
                ):
                    return cell

    # Check for draw (board full)
    if all(board[0][col] is not None for col in range(COLUMNS)):
        return "draw"

    return None


def apply_move(board: Board, col: int, player: Player) -> Board:
    """Applies a move and returns a new board state."""
    if board[0][col] is not None:
        raise ValueError(f"Column {col} is full")

    new_board: Board = [row[:] for row in board]
    for row in range(ROWS - 1, -1, -1):
        if new_board[row][col] is None:
            new_board[row][col] = player
            break

    return new_board


def order_moves(moves: list[int]) -> list[int]:
    """Orders moves with center columns first for better alpha-beta pruning."""
    center_col = COLUMNS // 2
    return sorted(moves, key=lambda m: abs(m - center_col))


def minimax_search(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    player: Player,
    current_player: Player,
) -> tuple[float, int | None]:
    """Minimax search with alpha-beta pruning."""
    # Check terminal states
    winner = check_winner(board)
    if winner is not None:
        if winner == "draw":
            return 0.0, None
        win_score = EVAL_WEIGHTS["WIN"] + depth * 100
        return (win_score if winner == player else -win_score), None

    legal_moves = get_legal_moves(board)
    if not legal_moves:
        return 0.0, None

    # Depth limit reached
    if depth == 0:
        return float(evaluate_position(board, player)), None

    ordered_moves = order_moves(legal_moves)
    next_player: Player = 2 if current_player == 1 else 1

    if maximizing:
        max_score = float("-inf")
        best_move = ordered_moves[0]

        for move in ordered_moves:
            new_board = apply_move(board, move, current_player)
            score, _ = minimax_search(
                new_board, depth - 1, alpha, beta, False, player, next_player
            )

            if score > max_score:
                max_score = score
                best_move = move

            alpha = max(alpha, score)
            if beta <= alpha:
                break

        return max_score, best_move
    else:
        min_score = float("inf")
        best_move = ordered_moves[0]

        for move in ordered_moves:
            new_board = apply_move(board, move, current_player)
            score, _ = minimax_search(
                new_board, depth - 1, alpha, beta, True, player, next_player
            )

            if score < min_score:
                min_score = score
                best_move = move

            beta = min(beta, score)
            if beta <= alpha:
                break

        return min_score, best_move


class MinimaxAgent(Agent):
    """
    Minimax agent with alpha-beta pruning.

    Matches the behavior of the TypeScript minimax-engine.ts.
    """

    def __init__(self, depth: int, error_rate: float = 0.0, name_override: str | None = None):
        """
        Initialize minimax agent.

        Args:
            depth: Search depth (plies)
            error_rate: Probability of making a random move (0-1)
            name_override: Optional custom name
        """
        self._depth = depth
        self._error_rate = error_rate
        self._name_override = name_override

    @property
    def name(self) -> str:
        if self._name_override:
            return self._name_override
        return f"minimax-d{self._depth}"

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name,
            description=f"Minimax with depth={self._depth}, error_rate={self._error_rate}",
            expected_elo=self._estimate_elo(),
        )

    def _estimate_elo(self) -> int:
        """Rough ELO estimate based on depth and error rate."""
        # Base ELO increases with depth
        base = min(400 + self._depth * 200, 2200)
        # Error rate reduces effective ELO
        penalty = int(self._error_rate * 500)
        return max(0, base - penalty)

    def get_move(self, board: Board, to_move: Player) -> int:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Random error based on error rate
        if self._error_rate > 0 and random.random() < self._error_rate:
            return random.choice(legal_moves)

        # Single legal move
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Run minimax search
        _, best_move = minimax_search(
            board=board,
            depth=self._depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing=True,
            player=to_move,
            current_player=to_move,
        )

        return best_move if best_move is not None else legal_moves[0]


# ============================================================================
# NEURAL AGENT
# ============================================================================


class NeuralAgent(Agent):
    """
    Neural network agent using ONNX inference.

    Wraps a trained neural network model for evaluation.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        name_override: str | None = None,
    ):
        """
        Initialize neural agent.

        Args:
            model_path: Path to ONNX model file
            temperature: Sampling temperature (0 = argmax, >0 = stochastic)
            name_override: Optional custom name
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for NeuralAgent")

        self._model_path = model_path
        self._temperature = temperature
        self._name_override = name_override
        self._session = ort.InferenceSession(model_path)

        # Get model input name
        self._input_name = self._session.get_inputs()[0].name

    @property
    def name(self) -> str:
        if self._name_override:
            return self._name_override
        import os

        return os.path.splitext(os.path.basename(self._model_path))[0]

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name,
            description=f"Neural network from {self._model_path}",
            expected_elo=0,  # Unknown until evaluated
        )

    def get_move(self, board: Board, to_move: Player) -> int:
        # Encode board
        encoded = encode_onehot(board, to_move)

        # Run inference
        policy, _ = self._session.run(None, {self._input_name: encoded.reshape(1, -1)})
        policy = policy[0]

        # Mask illegal moves
        legal_moves = get_legal_moves(board)
        masked_policy = np.full(COLUMNS, float("-inf"))
        for col in legal_moves:
            masked_policy[col] = policy[col]

        # Sample or argmax
        if self._temperature == 0:
            return int(np.argmax(masked_policy))
        else:
            # Apply temperature and softmax
            scaled = masked_policy / self._temperature
            scaled = scaled - np.max(scaled)  # Numerical stability
            probs = np.exp(scaled)
            probs = probs / np.sum(probs)
            return int(np.random.choice(COLUMNS, p=probs))


# ============================================================================
# REFERENCE AGENTS
# ============================================================================

# Reference agents matching botPersonas.ts
# These provide known benchmarks for ELO measurement
REFERENCE_AGENTS: dict[str, Agent] = {
    "random": RandomAgent(),
    "rookie": MinimaxAgent(depth=2, error_rate=0.35, name_override="rookie"),
    "rusty": MinimaxAgent(depth=3, error_rate=0.25, name_override="rusty"),
    "blitz": MinimaxAgent(depth=4, error_rate=0.18, name_override="blitz"),
    "nova": MinimaxAgent(depth=4, error_rate=0.15, name_override="nova"),
    "neuron": MinimaxAgent(depth=5, error_rate=0.12, name_override="neuron"),
    "scholar": MinimaxAgent(depth=6, error_rate=0.08, name_override="scholar"),
    "viper": MinimaxAgent(depth=5, error_rate=0.10, name_override="viper"),
    "titan": MinimaxAgent(depth=7, error_rate=0.04, name_override="titan"),
    "sentinel": MinimaxAgent(depth=10, error_rate=0.01, name_override="sentinel"),
}

# Expected ELO ratings for reference agents (from botPersonas.ts)
REFERENCE_ELOS: dict[str, int] = {
    "random": 0,
    "rookie": 700,
    "rusty": 900,
    "blitz": 1000,
    "nova": 1100,
    "neuron": 1200,
    "scholar": 1350,
    "viper": 1250,
    "titan": 1550,
    "sentinel": 1800,
}
