"""Tests for data validation utilities."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.validation import (
    validate_board,
    validate_move,
    validate_position,
    validate_game_record,
    validate_dataset,
    find_duplicate_positions,
)
from data.dataset import GameRecord, Position
from data.encoding import ROWS, COLUMNS


class TestValidateBoard:
    """Test board validation."""

    def test_valid_empty_board(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        result = validate_board(board)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_valid_with_pieces(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1
        board[5][4] = 2
        board[4][3] = 1

        result = validate_board(board)
        assert result.is_valid

    def test_invalid_dimensions_rows(self):
        board = [[None] * COLUMNS for _ in range(5)]  # Wrong row count
        result = validate_board(board)
        assert not result.is_valid
        assert any("rows" in e.lower() for e in result.errors)

    def test_invalid_dimensions_cols(self):
        board = [[None] * 5 for _ in range(ROWS)]  # Wrong column count
        result = validate_board(board)
        assert not result.is_valid

    def test_invalid_cell_value(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 3  # Invalid player number

        result = validate_board(board)
        assert not result.is_valid
        assert any("invalid cell" in e.lower() for e in result.errors)

    def test_floating_piece(self):
        """Piece without support below."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[3][3] = 1  # No piece below

        result = validate_board(board)
        assert not result.is_valid
        assert any("floating" in e.lower() for e in result.errors)

    def test_invalid_piece_count(self):
        """P2 has more pieces than P1."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][0] = 2
        board[5][1] = 2

        result = validate_board(board)
        assert not result.is_valid
        assert any("piece count" in e.lower() for e in result.errors)

    def test_valid_piece_count_equal(self):
        """Equal pieces (P2 just moved)."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][0] = 1
        board[5][1] = 2

        result = validate_board(board)
        assert result.is_valid

    def test_valid_piece_count_p1_ahead(self):
        """P1 has one more (P1 just moved)."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][0] = 1
        board[5][1] = 2
        board[5][2] = 1

        result = validate_board(board)
        assert result.is_valid


class TestValidateMove:
    """Test move validation."""

    def test_valid_move_empty(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        result = validate_move(board, 3, 1)
        assert result.is_valid

    def test_invalid_column_negative(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        result = validate_move(board, -1, 1)
        assert not result.is_valid

    def test_invalid_column_too_high(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        result = validate_move(board, 7, 1)
        assert not result.is_valid

    def test_full_column(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        # Fill column 3
        for row in range(ROWS):
            board[row][3] = 1

        result = validate_move(board, 3, 2)
        assert not result.is_valid
        assert any("full" in e.lower() for e in result.errors)


class TestValidatePosition:
    """Test position validation."""

    def test_valid_position(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        pos = Position(
            board=board,
            to_move=1,
            move_played=3,
            result=1.0,
            moves_to_end=5,
        )

        result = validate_position(pos)
        assert result.is_valid

    def test_invalid_to_move(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        pos = Position(
            board=board,
            to_move=3,  # Invalid
            move_played=3,
            result=1.0,
            moves_to_end=5,
        )

        result = validate_position(pos)
        assert not result.is_valid

    def test_invalid_move_played(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        pos = Position(
            board=board,
            to_move=1,
            move_played=10,  # Invalid column
            result=1.0,
            moves_to_end=5,
        )

        result = validate_position(pos)
        assert not result.is_valid

    def test_invalid_result(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        pos = Position(
            board=board,
            to_move=1,
            move_played=3,
            result=2.0,  # Out of range
            moves_to_end=5,
        )

        result = validate_position(pos)
        assert not result.is_valid


class TestValidateGameRecord:
    """Test game record validation."""

    def test_valid_game(self):
        record = GameRecord.from_moves("test", [3, 4, 3, 4])
        result = validate_game_record(record)
        assert result.is_valid

    def test_empty_positions(self):
        record = GameRecord(game_id="empty", positions=[])
        result = validate_game_record(record)
        assert not result.is_valid
        assert any("no positions" in e.lower() for e in result.errors)

    def test_full_game_valid(self):
        """A complete winning game should be valid."""
        # P1 vertical win
        record = GameRecord.from_moves("win", [3, 4, 3, 4, 3, 4, 3])
        result = validate_game_record(record)
        assert result.is_valid


class TestValidateDataset:
    """Test dataset validation."""

    def test_all_valid(self):
        records = [
            GameRecord.from_moves("g1", [3, 4, 3]),
            GameRecord.from_moves("g2", [2, 3, 2, 3]),
        ]

        stats = validate_dataset(records)

        assert stats["total_games"] == 2
        assert stats["valid_games"] == 2
        assert stats["invalid_games"] == 0
        assert stats["validation_rate"] == 100.0


class TestFindDuplicates:
    """Test duplicate position detection."""

    def test_no_duplicates(self):
        records = [
            GameRecord.from_moves("g1", [3, 4, 3]),
            GameRecord.from_moves("g2", [0, 1, 2]),
        ]

        stats = find_duplicate_positions(records)

        # Different games, likely different positions
        assert stats["total_positions"] == 6

    def test_same_opening(self):
        """Same opening position appears multiple times."""
        records = [
            GameRecord.from_moves("g1", [3, 4, 3]),
            GameRecord.from_moves("g2", [3, 4, 2]),  # Same first 2 moves
        ]

        stats = find_duplicate_positions(records)

        # First two positions are identical
        assert stats["positions_appearing_multiple"] >= 1
