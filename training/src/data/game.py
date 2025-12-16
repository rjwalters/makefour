"""
Connect Four Game Logic

Pure Python implementation of Connect Four rules for training data generation.
"""

from typing import Literal
from .encoding import Board, Player, ROWS, COLUMNS


def get_legal_moves(board: Board) -> list[int]:
    """
    Returns list of valid column indices where a piece can be dropped.
    A column is valid if the top row (row 0) is empty.
    """
    return [col for col in range(COLUMNS) if board[0][col] is None]


def make_move(board: Board, col: int, player: Player) -> Board:
    """
    Makes a move by dropping a piece in the specified column.
    Returns a new board (does not modify original).

    Raises:
        ValueError: If the column is full.
    """
    if board[0][col] is not None:
        raise ValueError(f"Column {col} is full")

    # Create a copy of the board
    new_board: Board = [row[:] for row in board]

    # Find the lowest empty row in the column
    for row in range(ROWS - 1, -1, -1):
        if new_board[row][col] is None:
            new_board[row][col] = player
            break

    return new_board


def check_winner(board: Board) -> Player | None:
    """
    Checks if there's a winner on the board.

    Returns:
        1 if player 1 wins, 2 if player 2 wins, None if no winner yet.
    """
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

    return None


def is_board_full(board: Board) -> bool:
    """Checks if the board is completely full (draw)."""
    return all(board[0][col] is not None for col in range(COLUMNS))


def get_game_result(board: Board) -> Literal["player1_win", "player2_win", "draw", "ongoing"]:
    """
    Determines the current game state.

    Returns:
        "player1_win": Player 1 has won
        "player2_win": Player 2 has won
        "draw": Board is full with no winner
        "ongoing": Game is still in progress
    """
    winner = check_winner(board)
    if winner == 1:
        return "player1_win"
    elif winner == 2:
        return "player2_win"
    elif is_board_full(board):
        return "draw"
    else:
        return "ongoing"


class ConnectFourGame:
    """
    Manages a Connect Four game state.
    """

    def __init__(self):
        self.board: Board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
        self.current_player: Player = 1
        self.move_history: list[int] = []
        self.winner: Player | None = None
        self.is_draw: bool = False

    def reset(self) -> None:
        """Resets the game to initial state."""
        self.board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
        self.current_player = 1
        self.move_history = []
        self.winner = None
        self.is_draw = False

    def get_legal_moves(self) -> list[int]:
        """Returns list of valid moves."""
        return get_legal_moves(self.board)

    def make_move(self, col: int) -> bool:
        """
        Makes a move in the specified column.

        Returns:
            True if the move was made successfully, False if invalid.
        """
        if col not in self.get_legal_moves():
            return False

        if self.is_terminal():
            return False

        self.board = make_move(self.board, col, self.current_player)
        self.move_history.append(col)

        # Check for winner
        self.winner = check_winner(self.board)
        if self.winner is None and is_board_full(self.board):
            self.is_draw = True

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return True

    def is_terminal(self) -> bool:
        """Returns True if the game is over."""
        return self.winner is not None or self.is_draw

    def get_result(self) -> float:
        """
        Returns the game result from player 1's perspective.

        Returns:
            1.0 if player 1 wins, -1.0 if player 2 wins, 0.0 for draw.
        """
        if self.winner == 1:
            return 1.0
        elif self.winner == 2:
            return -1.0
        else:
            return 0.0

    def get_result_for_player(self, player: Player) -> float:
        """
        Returns the game result from the specified player's perspective.
        """
        result = self.get_result()
        return result if player == 1 else -result

    def copy(self) -> "ConnectFourGame":
        """Returns a deep copy of the game."""
        game = ConnectFourGame()
        game.board = [row[:] for row in self.board]
        game.current_player = self.current_player
        game.move_history = self.move_history[:]
        game.winner = self.winner
        game.is_draw = self.is_draw
        return game

    @classmethod
    def from_moves(cls, moves: list[int]) -> "ConnectFourGame":
        """Creates a game from a sequence of moves."""
        game = cls()
        for move in moves:
            if not game.make_move(move):
                raise ValueError(f"Invalid move {move} at position {len(game.move_history)}")
        return game

    def __str__(self) -> str:
        """Returns a string representation of the board."""
        symbols = {None: ".", 1: "X", 2: "O"}
        lines = []
        for row in self.board:
            lines.append(" ".join(symbols[cell] for cell in row))
        lines.append("0 1 2 3 4 5 6")
        lines.append(f"Current player: {self.current_player}")
        if self.winner:
            lines.append(f"Winner: Player {self.winner}")
        elif self.is_draw:
            lines.append("Result: Draw")
        return "\n".join(lines)
