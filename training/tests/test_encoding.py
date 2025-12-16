"""
Tests for position encoding functions.

These tests verify that the Python encodings match the TypeScript implementation
in src/ai/neural/encoding.ts exactly.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.encoding import (
    ROWS,
    COLUMNS,
    encode_onehot,
    encode_onehot_3d,
    encode_flat_binary,
    encode_bitboard,
    decode_onehot,
    encode_position,
    get_input_shape,
    get_input_size,
    board_from_moves,
    flip_board_horizontal,
    flip_move_horizontal,
)


class TestConstants:
    """Test board dimension constants."""

    def test_rows(self):
        assert ROWS == 6

    def test_columns(self):
        assert COLUMNS == 7


class TestEncodeOnehot:
    """Test one-hot encoding (must match TypeScript encodeOneHot)."""

    def test_empty_board_player1(self):
        """Empty board with player 1 to move."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_onehot(board, 1)

        assert encoded.dtype == np.float32
        assert len(encoded) == 126  # 3 * 6 * 7

        # Channel 0 (player 1 pieces): all zeros
        assert np.sum(encoded[:42]) == 0

        # Channel 1 (player 2 pieces): all zeros
        assert np.sum(encoded[42:84]) == 0

        # Channel 2 (current player): all ones (player 1 to move)
        assert np.sum(encoded[84:126]) == 42

    def test_empty_board_player2(self):
        """Empty board with player 2 to move."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_onehot(board, 2)

        # Channel 2: all zeros (player 2 to move)
        assert np.sum(encoded[84:126]) == 0

    def test_single_piece(self):
        """Board with a single piece in bottom center."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1  # Player 1 piece at row 5, col 3

        encoded = encode_onehot(board, 2)

        # Channel 0: one piece at position 5*7+3 = 38
        assert encoded[38] == 1
        assert np.sum(encoded[:42]) == 1

        # Channel 1: no player 2 pieces
        assert np.sum(encoded[42:84]) == 0

    def test_multiple_pieces(self):
        """Board with pieces from both players."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1  # P1 at bottom center
        board[5][4] = 2  # P2 next to it
        board[4][3] = 1  # P1 on top of first piece

        encoded = encode_onehot(board, 2)

        # Channel 0: P1 pieces at (5,3) and (4,3)
        assert encoded[5 * 7 + 3] == 1  # 38
        assert encoded[4 * 7 + 3] == 1  # 31
        assert np.sum(encoded[:42]) == 2

        # Channel 1: P2 piece at (5,4)
        assert encoded[42 + 5 * 7 + 4] == 1  # 42 + 39 = 81
        assert np.sum(encoded[42:84]) == 1

    def test_encode_onehot_3d_shape(self):
        """Test 3D encoding has correct shape."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_onehot_3d(board, 1)

        assert encoded.shape == (3, ROWS, COLUMNS)
        assert encoded.shape == (3, 6, 7)


class TestEncodeFlatBinary:
    """Test flat binary encoding (must match TypeScript encodeFlatBinary)."""

    def test_empty_board_player1(self):
        """Empty board with player 1 to move."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_flat_binary(board, 1)

        assert encoded.dtype == np.float32
        assert len(encoded) == 85  # 42 + 42 + 1

        # First 42: player 1 pieces (all zeros)
        assert np.sum(encoded[:42]) == 0

        # Next 42: player 2 pieces (all zeros)
        assert np.sum(encoded[42:84]) == 0

        # Last bit: current player (1 for player 1)
        assert encoded[84] == 1

    def test_empty_board_player2(self):
        """Empty board with player 2 to move."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_flat_binary(board, 2)

        # Last bit: current player (0 for player 2)
        assert encoded[84] == 0

    def test_single_piece(self):
        """Board with a single piece."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1  # Player 1 piece

        encoded = encode_flat_binary(board, 1)

        # Player 1 piece at row 5, col 3 -> index 5*7+3 = 38
        assert encoded[38] == 1
        assert np.sum(encoded[:42]) == 1

    def test_multiple_pieces(self):
        """Board with multiple pieces."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1
        board[5][4] = 2
        board[4][3] = 1

        encoded = encode_flat_binary(board, 2)

        # P1 pieces
        assert encoded[5 * 7 + 3] == 1
        assert encoded[4 * 7 + 3] == 1
        assert np.sum(encoded[:42]) == 2

        # P2 piece
        assert encoded[42 + 5 * 7 + 4] == 1
        assert np.sum(encoded[42:84]) == 1


class TestEncodeBitboard:
    """Test bitboard encoding (must match TypeScript encodeBitboard)."""

    def test_empty_board(self):
        """Empty board encoding."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_bitboard(board, 1, move_count=0)

        assert encoded.dtype == np.float32
        assert len(encoded) == 4

        assert encoded[0] == 0  # p1_bits
        assert encoded[1] == 0  # p2_bits
        assert encoded[2] == 1  # player_bit (player 1)
        assert encoded[3] == 0  # move_count

    def test_single_piece(self):
        """Board with single piece."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1  # Row 5, Col 3 -> bit 38

        encoded = encode_bitboard(board, 2, move_count=1)

        expected_p1_bits = 1 << 38
        assert encoded[0] == expected_p1_bits
        assert encoded[1] == 0
        assert encoded[2] == 0  # player 2 to move
        assert encoded[3] == 1  # 1 move made

    def test_multiple_pieces(self):
        """Board with multiple pieces."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][0] = 1  # bit 35
        board[5][6] = 2  # bit 41

        encoded = encode_bitboard(board, 1, move_count=2)

        expected_p1_bits = 1 << 35
        expected_p2_bits = 1 << 41

        assert encoded[0] == expected_p1_bits
        assert encoded[1] == expected_p2_bits


class TestDecodeOnehot:
    """Test decoding one-hot back to board."""

    def test_roundtrip_empty(self):
        """Encode then decode empty board."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_onehot(board, 1)
        decoded_board, decoded_player = decode_onehot(encoded)

        assert decoded_board == board
        assert decoded_player == 1

    def test_roundtrip_with_pieces(self):
        """Encode then decode board with pieces."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1
        board[5][4] = 2
        board[4][3] = 1

        encoded = encode_onehot(board, 2)
        decoded_board, decoded_player = decode_onehot(encoded)

        assert decoded_board == board
        assert decoded_player == 2

    def test_decode_3d_input(self):
        """Decode 3D shaped input."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1

        encoded_3d = encode_onehot_3d(board, 1)
        decoded_board, decoded_player = decode_onehot(encoded_3d)

        assert decoded_board == board
        assert decoded_player == 1


class TestEncodePosition:
    """Test unified encode_position function."""

    def test_onehot_encoding(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_position(board, 1, "onehot-6x7x3")
        assert len(encoded) == 126

    def test_flat_binary_encoding(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_position(board, 1, "flat-binary")
        assert len(encoded) == 85

    def test_bitboard_encoding(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_position(board, 1, "bitboard")
        assert len(encoded) == 4

    def test_invalid_encoding(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        with pytest.raises(ValueError, match="Unknown encoding"):
            encode_position(board, 1, "invalid")


class TestGetInputShape:
    """Test input shape utilities."""

    def test_onehot_shape(self):
        shape = get_input_shape("onehot-6x7x3")
        assert shape == (1, 3, 6, 7)

    def test_flat_binary_shape(self):
        shape = get_input_shape("flat-binary")
        assert shape == (1, 85)

    def test_bitboard_shape(self):
        shape = get_input_shape("bitboard")
        assert shape == (1, 4)

    def test_onehot_size(self):
        size = get_input_size("onehot-6x7x3")
        assert size == 126

    def test_flat_binary_size(self):
        size = get_input_size("flat-binary")
        assert size == 85


class TestBoardFromMoves:
    """Test reconstructing board from move sequence."""

    def test_empty_moves(self):
        """No moves = empty board, player 1 to move."""
        board, player = board_from_moves([])
        assert board == [[None] * COLUMNS for _ in range(ROWS)]
        assert player == 1

    def test_single_move(self):
        """Single move in center column."""
        board, player = board_from_moves([3])

        # Piece at bottom center
        assert board[5][3] == 1
        # Player 2 to move next
        assert player == 2

    def test_two_moves(self):
        """Two moves, pieces stack correctly."""
        board, player = board_from_moves([3, 3])

        # First piece at bottom
        assert board[5][3] == 1
        # Second piece on top
        assert board[4][3] == 2
        # Player 1 to move next
        assert player == 1

    def test_full_game(self):
        """Verify a full game's board state."""
        # Quick vertical win for player 1
        moves = [3, 4, 3, 4, 3, 4, 3]
        board, player = board_from_moves(moves)

        # Player 1 has vertical four in column 3
        assert board[5][3] == 1
        assert board[4][3] == 1
        assert board[3][3] == 1
        assert board[2][3] == 1


class TestFlipBoard:
    """Test horizontal flip for data augmentation."""

    def test_flip_empty(self):
        """Flipping empty board gives empty board."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        flipped = flip_board_horizontal(board)
        assert flipped == board

    def test_flip_single_piece(self):
        """Flip single piece from col 0 to col 6."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][0] = 1

        flipped = flip_board_horizontal(board)

        assert flipped[5][0] is None
        assert flipped[5][6] == 1

    def test_flip_center(self):
        """Center column (3) stays in place."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][3] = 1

        flipped = flip_board_horizontal(board)

        assert flipped[5][3] == 1

    def test_flip_move(self):
        """Test move index flipping."""
        assert flip_move_horizontal(0) == 6
        assert flip_move_horizontal(1) == 5
        assert flip_move_horizontal(2) == 4
        assert flip_move_horizontal(3) == 3  # Center stays
        assert flip_move_horizontal(4) == 2
        assert flip_move_horizontal(5) == 1
        assert flip_move_horizontal(6) == 0

    def test_double_flip(self):
        """Double flip returns original."""
        board = [[None] * COLUMNS for _ in range(ROWS)]
        board[5][1] = 1
        board[4][5] = 2

        double_flipped = flip_board_horizontal(flip_board_horizontal(board))

        assert double_flipped == board


class TestEncodingParity:
    """
    These tests verify encoding parity with TypeScript.

    The expected values are derived from running the TypeScript encoders
    on the same board positions.
    """

    def test_parity_case_1_empty_board(self):
        """
        Empty board, player 1 to move.

        TypeScript: encodeOneHot() returns 126-element Float32Array
        - First 84 elements: all 0
        - Last 42 elements: all 1
        """
        board = [[None] * COLUMNS for _ in range(ROWS)]
        encoded = encode_onehot(board, 1)

        # Verify structure matches TypeScript
        assert np.all(encoded[:84] == 0)
        assert np.all(encoded[84:126] == 1)

    def test_parity_case_2_simple_position(self):
        """
        Position after moves [3, 4, 3].

        Board:
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . X . . .   <- P1 at (4,3)
        . . . X O . .   <- P1 at (5,3), P2 at (5,4)

        Player 2 to move (3 moves made, so player 2's turn)
        """
        moves = [3, 4, 3]
        board, player = board_from_moves(moves)

        # Verify board state
        assert board[5][3] == 1
        assert board[5][4] == 2
        assert board[4][3] == 1
        assert player == 2

        # One-hot encoding
        onehot = encode_onehot(board, player)

        # Channel 0 (P1 pieces): positions 38 and 31 should be 1
        assert onehot[5 * 7 + 3] == 1  # (5,3) -> 38
        assert onehot[4 * 7 + 3] == 1  # (4,3) -> 31
        assert np.sum(onehot[:42]) == 2

        # Channel 1 (P2 pieces): position 39 should be 1
        assert onehot[42 + 5 * 7 + 4] == 1  # 42 + 39 = 81
        assert np.sum(onehot[42:84]) == 1

        # Channel 2: all 0 (player 2 to move)
        assert np.sum(onehot[84:126]) == 0

        # Flat binary encoding
        flat = encode_flat_binary(board, player)
        assert flat[5 * 7 + 3] == 1  # P1 at (5,3)
        assert flat[4 * 7 + 3] == 1  # P1 at (4,3)
        assert flat[42 + 5 * 7 + 4] == 1  # P2 at (5,4)
        assert flat[84] == 0  # Player 2 to move

    def test_parity_case_3_first_move(self):
        """
        Position after first move in center.

        TypeScript expected behavior:
        - One piece at row 5 (bottom), column 3
        - Player 2 to move
        """
        board, player = board_from_moves([3])

        assert player == 2

        # Verify one-hot encoding
        onehot = encode_onehot(board, player)

        # Only one piece: P1 at (5,3)
        assert np.sum(onehot[:42]) == 1
        assert onehot[5 * 7 + 3] == 1

        # No P2 pieces
        assert np.sum(onehot[42:84]) == 0

        # Player 2 to move
        assert np.sum(onehot[84:126]) == 0
