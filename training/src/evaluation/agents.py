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

from ..data import COLUMNS, ROWS, Board, Player, encode_onehot, encode_flat_binary

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

    def __init__(
        self,
        depth: int,
        error_rate: float = 0.0,
        blunder_rate: float = 0.0,
        name_override: str | None = None,
        max_depth: int | None = None,
    ):
        """
        Initialize minimax agent.

        Args:
            depth: Search depth (plies)
            error_rate: Probability of making a random move (0-1)
            blunder_rate: Probability of missing a winning/blocking move (0-1)
            name_override: Optional custom name
            max_depth: Maximum depth cap (for fast tournament mode)
        """
        self._depth = depth
        self._error_rate = error_rate
        self._blunder_rate = blunder_rate
        self._name_override = name_override
        self._max_depth = max_depth

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

        # Random error based on error rate - completely random move
        if self._error_rate > 0 and random.random() < self._error_rate:
            return random.choice(legal_moves)

        # Single legal move
        if len(legal_moves) == 1:
            return legal_moves[0]

        opponent: Player = 2 if to_move == 1 else 1

        # Check for immediate winning move
        winning_move = None
        for move in legal_moves:
            new_board = apply_move(board, move, to_move)
            if check_winner(new_board) == to_move:
                winning_move = move
                break

        # Blunder: miss the winning move
        if winning_move is not None:
            if self._blunder_rate > 0 and random.random() < self._blunder_rate:
                # Miss the win, pick something else
                other_moves = [m for m in legal_moves if m != winning_move]
                if other_moves:
                    return random.choice(other_moves)
            return winning_move

        # Check for blocking move (opponent would win)
        blocking_move = None
        for move in legal_moves:
            new_board = apply_move(board, move, opponent)
            if check_winner(new_board) == opponent:
                blocking_move = move
                break

        # Blunder: miss the block
        if blocking_move is not None:
            if self._blunder_rate > 0 and random.random() < self._blunder_rate:
                # Miss the block, pick something else
                other_moves = [m for m in legal_moves if m != blocking_move]
                if other_moves:
                    return random.choice(other_moves)
            return blocking_move

        # Apply depth cap if set
        effective_depth = self._depth
        if self._max_depth is not None:
            effective_depth = min(self._depth, self._max_depth)

        # Run minimax search
        _, best_move = minimax_search(
            board=board,
            depth=effective_depth,
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

        # Get model input name and shape
        self._input_name = self._session.get_inputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        # Detect input size (second dimension, first is batch)
        self._input_size = input_shape[1] if len(input_shape) > 1 else 85

        # Auto-detect encoding based on input size
        if self._input_size == 85:
            self._encoding = "flat-binary"
        else:  # 126 or other
            self._encoding = "onehot"

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
        # Encode board using appropriate encoding
        if self._encoding == "flat-binary":
            encoded = encode_flat_binary(board, to_move)
        else:
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
# DEEP MINIMAX AGENT (for Oracle)
# ============================================================================


class DeepMinimaxAgent(Agent):
    """
    Deep minimax agent with transposition table for full-depth search.

    Used for Oracle bot which plays perfectly with depth 42.
    """

    def __init__(self, depth: int = 42, name_override: str | None = None):
        self._depth = depth
        self._name_override = name_override
        self._transposition_table: dict[str, tuple[float, int | None]] = {}

    @property
    def name(self) -> str:
        return self._name_override or f"deep-minimax-d{self._depth}"

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name,
            description=f"Deep minimax with depth={self._depth} and transposition table",
            expected_elo=2200,
        )

    def reset(self) -> None:
        self._transposition_table.clear()

    def _board_hash(self, board: Board, to_move: Player) -> str:
        """Create hash key for board position."""
        cells = []
        for row in board:
            for cell in row:
                cells.append("0" if cell is None else str(cell))
        return "".join(cells) + str(to_move)

    def get_move(self, board: Board, to_move: Player) -> int:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Deep search with transposition table
        _, best_move = self._deep_search(
            board=board,
            depth=min(self._depth, 42),  # Cap at 42 (max moves)
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing=True,
            player=to_move,
            current_player=to_move,
        )

        return best_move if best_move is not None else legal_moves[0]

    def _deep_search(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        player: Player,
        current_player: Player,
    ) -> tuple[float, int | None]:
        """Deep minimax with transposition table."""
        # Check transposition table
        board_key = self._board_hash(board, current_player)
        if board_key in self._transposition_table:
            return self._transposition_table[board_key]

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

        if depth == 0:
            return float(evaluate_position(board, player)), None

        ordered_moves = order_moves(legal_moves)
        next_player: Player = 2 if current_player == 1 else 1

        if maximizing:
            max_score = float("-inf")
            best_move = ordered_moves[0]

            for move in ordered_moves:
                new_board = apply_move(board, move, current_player)
                score, _ = self._deep_search(
                    new_board, depth - 1, alpha, beta, False, player, next_player
                )

                if score > max_score:
                    max_score = score
                    best_move = move

                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            result = (max_score, best_move)
        else:
            min_score = float("inf")
            best_move = ordered_moves[0]

            for move in ordered_moves:
                new_board = apply_move(board, move, current_player)
                score, _ = self._deep_search(
                    new_board, depth - 1, alpha, beta, True, player, next_player
                )

                if score < min_score:
                    min_score = score
                    best_move = move

                beta = min(beta, score)
                if beta <= alpha:
                    break

            result = (min_score, best_move)

        # Store in transposition table
        self._transposition_table[board_key] = result
        return result


# ============================================================================
# 2SWAP STRATEGY AGENTS
# ============================================================================


class ClaimEvenAgent(Agent):
    """
    Implements claimeven strategy from 2swap.

    The strategy: respond directly above opponent's moves to claim even rows.
    Works best as Player 2 (Yellow) who plays the last move.
    """

    def __init__(self, error_rate: float = 0.05, name_override: str | None = None):
        self._error_rate = error_rate
        self._name_override = name_override

    @property
    def name(self) -> str:
        return self._name_override or "claimeven"

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name,
            description="2swap claimeven strategy - claims even rows",
            expected_elo=1150,
        )

    def get_move(self, board: Board, to_move: Player) -> int:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Random error
        if self._error_rate > 0 and random.random() < self._error_rate:
            return random.choice(legal_moves)

        # First move: center
        if all(board[ROWS - 1][col] is None for col in range(COLUMNS)):
            return COLUMNS // 2

        # Find opponent's last move by comparing with previous state
        # For simplicity, look for columns with recent activity
        opponent: Player = 2 if to_move == 1 else 1

        # Strategy: play in same column as opponent if possible (claimeven)
        # Otherwise fall back to evaluation
        for col in legal_moves:
            # Check if opponent played in this column recently (has their piece)
            has_opponent = False
            for row in range(ROWS):
                if board[row][col] == opponent:
                    has_opponent = True
                    break
            if has_opponent and col in legal_moves:
                # Play above if possible
                return col

        # Fallback: prefer center columns
        center_preference = order_moves(legal_moves)

        # Check for immediate wins/blocks
        for move in center_preference:
            new_board = apply_move(board, move, to_move)
            if check_winner(new_board) == to_move:
                return move

        for move in center_preference:
            new_board = apply_move(board, move, opponent)
            if check_winner(new_board) == opponent:
                return move

        return center_preference[0]


class ParityAgent(Agent):
    """
    Implements parity strategy from 2swap.

    Red (player 1) prioritizes odd-row threats.
    Yellow (player 2) prioritizes even-row threats.
    """

    def __init__(self, error_rate: float = 0.05, name_override: str | None = None):
        self._error_rate = error_rate
        self._name_override = name_override

    @property
    def name(self) -> str:
        return self._name_override or "parity"

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name,
            description="2swap parity strategy - prioritizes favorable row threats",
            expected_elo=1200,
        )

    def _get_landing_row(self, board: Board, col: int) -> int | None:
        """Get the row where a piece would land in the given column."""
        for row in range(ROWS - 1, -1, -1):
            if board[row][col] is None:
                return row
        return None

    def _count_threats_on_row(self, board: Board, player: Player, target_row: int) -> int:
        """Count potential winning threats on a specific row."""
        count = 0
        for col in range(COLUMNS - 3):
            window = [board[target_row][col + i] for i in range(4)]
            player_count = sum(1 for c in window if c == player)
            empty_count = sum(1 for c in window if c is None)
            opponent_count = 4 - player_count - empty_count

            if player_count >= 2 and opponent_count == 0 and empty_count > 0:
                count += 1
        return count

    def get_move(self, board: Board, to_move: Player) -> int:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Random error
        if self._error_rate > 0 and random.random() < self._error_rate:
            return random.choice(legal_moves)

        opponent: Player = 2 if to_move == 1 else 1

        # Check for immediate wins/blocks
        for move in legal_moves:
            new_board = apply_move(board, move, to_move)
            if check_winner(new_board) == to_move:
                return move

        for move in legal_moves:
            new_board = apply_move(board, move, opponent)
            if check_winner(new_board) == opponent:
                return move

        # Parity strategy: Red wants odd rows (0, 2, 4), Yellow wants even rows (1, 3, 5)
        # Row numbering: 0 is top, 5 is bottom
        # Odd-indexed rows from bottom: rows 5, 3, 1 (indices)
        # Player 1 (Red) wants threats on odd rows (1, 3, 5 from top = indices 1, 3, 5)
        # Player 2 (Yellow) wants threats on even rows (0, 2, 4 from top = indices 0, 2, 4)

        favored_parity = 1 if to_move == 1 else 0  # 1=odd indices, 0=even indices

        best_move = legal_moves[0]
        best_score = float("-inf")

        for move in legal_moves:
            landing_row = self._get_landing_row(board, move)
            if landing_row is None:
                continue

            score = 0
            # Favor landing on our preferred parity
            if landing_row % 2 == favored_parity:
                score += 50
            else:
                score -= 30

            # Bonus for center control
            score += (3 - abs(move - 3)) * 5

            # Evaluate the move with minimax-style heuristic
            new_board = apply_move(board, move, to_move)
            score += evaluate_position(new_board, to_move) / 100

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


class ThreatPairsAgent(Agent):
    """
    Implements threat pairs strategy from 2swap.

    Creates double threats (stacked or 7-shaped patterns) for combinatoric wins.
    """

    def __init__(self, error_rate: float = 0.05, name_override: str | None = None):
        self._error_rate = error_rate
        self._name_override = name_override

    @property
    def name(self) -> str:
        return self._name_override or "threat-pairs"

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name,
            description="2swap threat pairs strategy - creates double threats",
            expected_elo=1250,
        )

    def _get_landing_row(self, board: Board, col: int) -> int | None:
        """Get the row where a piece would land in the given column."""
        for row in range(ROWS - 1, -1, -1):
            if board[row][col] is None:
                return row
        return None

    def _count_threats(self, board: Board, player: Player) -> int:
        """Count the number of 3-in-a-row threats with empty 4th cell."""
        threats = 0

        # Horizontal threats
        for row in range(ROWS):
            for col in range(COLUMNS - 3):
                window = [board[row][col + i] for i in range(4)]
                player_count = sum(1 for c in window if c == player)
                empty_count = sum(1 for c in window if c is None)
                if player_count == 3 and empty_count == 1:
                    threats += 1

        # Vertical threats
        for col in range(COLUMNS):
            for row in range(ROWS - 3):
                window = [board[row + i][col] for i in range(4)]
                player_count = sum(1 for c in window if c == player)
                empty_count = sum(1 for c in window if c is None)
                if player_count == 3 and empty_count == 1:
                    threats += 1

        # Diagonal threats
        for row in range(ROWS - 3):
            for col in range(COLUMNS - 3):
                window = [board[row + i][col + i] for i in range(4)]
                player_count = sum(1 for c in window if c == player)
                empty_count = sum(1 for c in window if c is None)
                if player_count == 3 and empty_count == 1:
                    threats += 1

        for row in range(3, ROWS):
            for col in range(COLUMNS - 3):
                window = [board[row - i][col + i] for i in range(4)]
                player_count = sum(1 for c in window if c == player)
                empty_count = sum(1 for c in window if c is None)
                if player_count == 3 and empty_count == 1:
                    threats += 1

        return threats

    def get_move(self, board: Board, to_move: Player) -> int:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Random error
        if self._error_rate > 0 and random.random() < self._error_rate:
            return random.choice(legal_moves)

        opponent: Player = 2 if to_move == 1 else 1

        # Check for immediate wins
        for move in legal_moves:
            new_board = apply_move(board, move, to_move)
            if check_winner(new_board) == to_move:
                return move

        # Block opponent wins
        for move in legal_moves:
            new_board = apply_move(board, move, opponent)
            if check_winner(new_board) == opponent:
                return move

        # Find move that maximizes threats
        best_move = legal_moves[0]
        best_threat_count = -1

        for move in legal_moves:
            new_board = apply_move(board, move, to_move)
            threat_count = self._count_threats(new_board, to_move)

            # Bonus for creating stacked threats (7-shape)
            landing_row = self._get_landing_row(board, move)
            if landing_row is not None and landing_row > 0:
                # Check if playing above creates threat
                above_board = [row[:] for row in new_board]
                if above_board[landing_row - 1][move] is None:
                    above_board[landing_row - 1][move] = to_move
                    above_threats = self._count_threats(above_board, to_move)
                    threat_count += above_threats * 0.5  # Bonus for setup

            # Center bonus
            threat_count += (3 - abs(move - 3)) * 0.1

            if threat_count > best_threat_count:
                best_threat_count = threat_count
                best_move = move

        return best_move


# ============================================================================
# REFERENCE AGENTS
# ============================================================================

# Reference agents matching botPersonas.ts
# These provide known benchmarks for ELO measurement
REFERENCE_AGENTS: dict[str, Agent] = {
    "random": RandomAgent(),
    # Reduced depths and added blunder rates to weaken minimax bots
    # blunder_rate = probability of missing a winning or blocking move
    "rookie": MinimaxAgent(depth=1, error_rate=0.35, blunder_rate=0.15, name_override="rookie"),
    "rusty": MinimaxAgent(depth=2, error_rate=0.25, blunder_rate=0.12, name_override="rusty"),
    "blitz": MinimaxAgent(depth=3, error_rate=0.18, blunder_rate=0.10, name_override="blitz"),
    "nova": MinimaxAgent(depth=3, error_rate=0.15, blunder_rate=0.08, name_override="nova"),
    "neuron": MinimaxAgent(depth=4, error_rate=0.12, blunder_rate=0.06, name_override="neuron"),
    "scholar": MinimaxAgent(depth=5, error_rate=0.08, blunder_rate=0.04, name_override="scholar"),
    "viper": MinimaxAgent(depth=4, error_rate=0.10, blunder_rate=0.05, name_override="viper"),
    "titan": MinimaxAgent(depth=5, error_rate=0.04, blunder_rate=0.02, name_override="titan"),
    "sentinel": MinimaxAgent(depth=7, error_rate=0.01, blunder_rate=0.01, name_override="sentinel"),
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


def create_all_agents(
    models_dir: str | None = None,
    max_depth: int | None = None,
) -> dict[str, Agent]:
    """
    Create all 17 bot persona agents.

    Args:
        models_dir: Directory containing ONNX model files
        max_depth: Maximum search depth cap for minimax agents (for fast tournaments)

    Returns:
        Dictionary mapping agent ID to Agent instance
    """
    import os

    # Create reference agents with reduced depths and blunder rates
    agents: dict[str, Agent] = {
        "random": RandomAgent(),
        "rookie": MinimaxAgent(depth=1, error_rate=0.35, blunder_rate=0.15, name_override="rookie", max_depth=max_depth),
        "rusty": MinimaxAgent(depth=2, error_rate=0.25, blunder_rate=0.12, name_override="rusty", max_depth=max_depth),
        "blitz": MinimaxAgent(depth=3, error_rate=0.18, blunder_rate=0.10, name_override="blitz", max_depth=max_depth),
        "nova": MinimaxAgent(depth=3, error_rate=0.15, blunder_rate=0.08, name_override="nova", max_depth=max_depth),
        "neuron": MinimaxAgent(depth=4, error_rate=0.12, blunder_rate=0.06, name_override="neuron", max_depth=max_depth),
        "scholar": MinimaxAgent(depth=5, error_rate=0.08, blunder_rate=0.04, name_override="scholar", max_depth=max_depth),
        "viper": MinimaxAgent(depth=4, error_rate=0.10, blunder_rate=0.05, name_override="viper", max_depth=max_depth),
        "titan": MinimaxAgent(depth=5, error_rate=0.04, blunder_rate=0.02, name_override="titan", max_depth=max_depth),
        "sentinel": MinimaxAgent(depth=7, error_rate=0.01, blunder_rate=0.01, name_override="sentinel", max_depth=max_depth),
    }

    # Add Oracle (deep minimax)
    agents["oracle"] = DeepMinimaxAgent(depth=42, name_override="oracle")

    # Add 2swap strategy agents
    agents["2swap-claimeven"] = ClaimEvenAgent(error_rate=0.05, name_override="2swap-claimeven")
    agents["2swap-parity"] = ParityAgent(error_rate=0.05, name_override="2swap-parity")
    agents["2swap-threats"] = ThreatPairsAgent(error_rate=0.05, name_override="2swap-threats")

    # Add neural agents if models available
    if models_dir:
        try:
            # Neural Intuition - selfplay-v3
            model_path = os.path.join(models_dir, "selfplay-v3.onnx")
            if os.path.exists(model_path):
                agents["neural-intuition"] = NeuralAgent(
                    model_path=model_path,
                    temperature=0.5,
                    name_override="neural-intuition",
                )

            # Neural Spark - mlp-tiny-v1
            model_path = os.path.join(models_dir, "mlp-tiny-v1.onnx")
            if os.path.exists(model_path):
                agents["neural-spark"] = NeuralAgent(
                    model_path=model_path,
                    temperature=0.8,
                    name_override="neural-spark",
                )

            # Neural Echo - selfplay-v1
            model_path = os.path.join(models_dir, "selfplay-v1.onnx")
            if os.path.exists(model_path):
                agents["neural-echo"] = NeuralAgent(
                    model_path=model_path,
                    temperature=0.3,
                    name_override="neural-echo",
                )
        except ImportError:
            pass  # onnxruntime not available

    return agents


# All expected ELO ratings (from botPersonas.ts)
ALL_EXPECTED_ELOS: dict[str, int] = {
    **REFERENCE_ELOS,
    "oracle": 2200,
    "neural-intuition": 1000,
    "neural-spark": 800,
    "neural-echo": 900,
    "2swap-claimeven": 1150,
    "2swap-parity": 1200,
    "2swap-threats": 1250,
}
