"""
Data Validation Utilities

Validates game records and positions for training data quality.
"""

from dataclasses import dataclass
from typing import Literal

from .encoding import Board, Player, ROWS, COLUMNS
from .game import get_legal_moves, check_winner, make_move, ConnectFourGame
from .dataset import GameRecord, Position


@dataclass
class ValidationResult:
    """Result of validating a game or position."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    def __bool__(self) -> bool:
        return self.is_valid


def validate_board(board: Board) -> ValidationResult:
    """
    Validate that a board is a legal Connect Four position.

    Checks:
    - Board dimensions are correct
    - Only valid cell values (None, 1, 2)
    - Gravity is respected (no floating pieces)
    - Piece count difference is at most 1
    """
    errors = []
    warnings = []

    # Check dimensions
    if len(board) != ROWS:
        errors.append(f"Invalid number of rows: {len(board)} (expected {ROWS})")
        return ValidationResult(False, errors, warnings)

    for row_idx, row in enumerate(board):
        if len(row) != COLUMNS:
            errors.append(f"Row {row_idx} has invalid length: {len(row)} (expected {COLUMNS})")

    if errors:
        return ValidationResult(False, errors, warnings)

    # Check cell values
    for row_idx, row in enumerate(board):
        for col_idx, cell in enumerate(row):
            if cell not in (None, 1, 2):
                errors.append(f"Invalid cell value at ({row_idx}, {col_idx}): {cell}")

    if errors:
        return ValidationResult(False, errors, warnings)

    # Check gravity (pieces must stack from bottom)
    for col in range(COLUMNS):
        found_empty = False
        for row in range(ROWS):
            cell = board[row][col]
            if cell is None:
                found_empty = True
            elif found_empty:
                errors.append(f"Floating piece at ({row}, {col})")

    if errors:
        return ValidationResult(False, errors, warnings)

    # Check piece counts
    p1_count = sum(1 for row in board for cell in row if cell == 1)
    p2_count = sum(1 for row in board for cell in row if cell == 2)

    if abs(p1_count - p2_count) > 1:
        errors.append(f"Invalid piece count: P1={p1_count}, P2={p2_count}")
    elif p1_count < p2_count:
        errors.append(f"P2 has more pieces than P1 (impossible): P1={p1_count}, P2={p2_count}")

    if errors:
        return ValidationResult(False, errors, warnings)

    # Warning: check for games that continued after win
    winner = check_winner(board)
    if winner is not None:
        total_pieces = p1_count + p2_count
        # After someone wins, no more moves should be made
        # This is just a warning since some datasets might include post-win positions
        warnings.append(f"Board has a winner (Player {winner}) with {total_pieces} pieces")

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_move(board: Board, move: int, player: Player) -> ValidationResult:
    """
    Validate that a move is legal from the given position.
    """
    errors = []
    warnings = []

    # Check column is in range
    if not (0 <= move < COLUMNS):
        errors.append(f"Move {move} is out of range [0, {COLUMNS})")
        return ValidationResult(False, errors, warnings)

    # Check column is not full
    legal_moves = get_legal_moves(board)
    if move not in legal_moves:
        errors.append(f"Column {move} is full")

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_position(position: Position) -> ValidationResult:
    """
    Validate a single position.
    """
    errors = []
    warnings = []

    # Validate board
    board_result = validate_board(position.board)
    errors.extend(board_result.errors)
    warnings.extend(board_result.warnings)

    if not board_result.is_valid:
        return ValidationResult(False, errors, warnings)

    # Validate to_move
    if position.to_move not in (1, 2):
        errors.append(f"Invalid to_move: {position.to_move}")

    # Validate move_played
    if not (0 <= position.move_played < COLUMNS):
        errors.append(f"Invalid move_played: {position.move_played}")
    else:
        move_result = validate_move(position.board, position.move_played, position.to_move)
        errors.extend(move_result.errors)

    # Validate result
    if not (-1 <= position.result <= 1):
        errors.append(f"Invalid result: {position.result}")

    # Validate moves_to_end
    if position.moves_to_end < 0:
        errors.append(f"Invalid moves_to_end: {position.moves_to_end}")

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_game_record(record: GameRecord) -> ValidationResult:
    """
    Validate a complete game record.

    Checks:
    - All positions are valid
    - Moves form a valid game sequence
    - Results are consistent
    """
    errors = []
    warnings = []

    if not record.positions:
        errors.append("Game record has no positions")
        return ValidationResult(False, errors, warnings)

    # Validate each position
    for i, pos in enumerate(record.positions):
        pos_result = validate_position(pos)
        for error in pos_result.errors:
            errors.append(f"Position {i}: {error}")
        for warning in pos_result.warnings:
            warnings.append(f"Position {i}: {warning}")

    if errors:
        return ValidationResult(False, errors, warnings)

    # Replay the game to verify consistency
    game = ConnectFourGame()

    for i, pos in enumerate(record.positions):
        # Check board matches expected state
        expected_board = game.board
        if pos.board != expected_board:
            errors.append(f"Position {i}: Board state mismatch")

        # Check current player
        if pos.to_move != game.current_player:
            errors.append(f"Position {i}: Expected player {game.current_player}, got {pos.to_move}")

        # Make the move
        if not game.make_move(pos.move_played):
            errors.append(f"Position {i}: Invalid move {pos.move_played}")
            break

    # Check final result consistency
    if game.is_terminal():
        expected_result = game.get_result()
        # First position's result should match (from player 1's perspective)
        if record.positions[0].result != expected_result:
            warnings.append(
                f"Result mismatch: first position has {record.positions[0].result}, "
                f"game result is {expected_result}"
            )

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_dataset(records: list[GameRecord]) -> dict:
    """
    Validate an entire dataset and return summary statistics.

    Args:
        records: List of game records to validate.

    Returns:
        Dictionary with validation results.
    """
    total = len(records)
    valid = 0
    invalid = 0
    all_errors: list[tuple[str, list[str]]] = []
    all_warnings: list[tuple[str, list[str]]] = []

    for record in records:
        result = validate_game_record(record)

        if result.is_valid:
            valid += 1
        else:
            invalid += 1
            all_errors.append((record.game_id, result.errors))

        if result.warnings:
            all_warnings.append((record.game_id, result.warnings))

    return {
        "total_games": total,
        "valid_games": valid,
        "invalid_games": invalid,
        "validation_rate": round(valid / total * 100, 2) if total > 0 else 0,
        "errors": all_errors[:10],  # First 10 errors
        "warnings": all_warnings[:10],  # First 10 warnings
        "total_errors": len(all_errors),
        "total_warnings": len(all_warnings),
    }


def find_duplicate_positions(records: list[GameRecord]) -> dict:
    """
    Find duplicate positions across a dataset.

    Returns:
        Dictionary with duplicate statistics.
    """
    position_hashes: dict[str, int] = {}

    for record in records:
        for pos in record.positions:
            # Create a simple hash of the board state
            board_str = str(pos.board) + str(pos.to_move)
            position_hashes[board_str] = position_hashes.get(board_str, 0) + 1

    unique = sum(1 for count in position_hashes.values() if count == 1)
    duplicates = sum(1 for count in position_hashes.values() if count > 1)
    total = sum(position_hashes.values())

    return {
        "total_positions": total,
        "unique_positions": len(position_hashes),
        "positions_appearing_once": unique,
        "positions_appearing_multiple": duplicates,
        "duplication_rate": round((1 - len(position_hashes) / total) * 100, 2) if total > 0 else 0,
    }
