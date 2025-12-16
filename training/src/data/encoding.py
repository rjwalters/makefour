"""
Position Encoding for Neural Network Input

Provides different encoding schemes for representing Connect Four positions
as neural network inputs. These MUST match the TypeScript encodings in
src/ai/neural/encoding.ts exactly.
"""

from typing import Literal, TypeAlias
import numpy as np

# Board dimensions (must match TypeScript)
ROWS = 6
COLUMNS = 7

# Type aliases
Board: TypeAlias = list[list[int | None]]  # 6x7, None=empty, 1=player1, 2=player2
Player: TypeAlias = Literal[1, 2]
EncodingType: TypeAlias = Literal["onehot-6x7x3", "flat-binary", "bitboard"]


def encode_onehot(board: Board, current_player: Player) -> np.ndarray:
    """
    Encodes a position using one-hot encoding with 3 channels.
    Shape: [3, 6, 7] = [channels, rows, columns]

    Channel 0: Player 1's pieces (1 where P1 has a piece, 0 elsewhere)
    Channel 1: Player 2's pieces (1 where P2 has a piece, 0 elsewhere)
    Channel 2: Current player indicator (all 1s if P1 to move, all 0s if P2)

    This encoding preserves spatial relationships for CNNs.

    IMPORTANT: This must match encodeOneHot() in src/ai/neural/encoding.ts
    """
    channel_size = ROWS * COLUMNS  # 42
    total_size = 3 * channel_size  # 126

    # Create flat array (matches TypeScript Float32Array layout)
    encoded = np.zeros(total_size, dtype=np.float32)

    # Channel 0: Player 1 pieces
    for row in range(ROWS):
        for col in range(COLUMNS):
            if board[row][col] == 1:
                encoded[row * COLUMNS + col] = 1.0

    # Channel 1: Player 2 pieces
    for row in range(ROWS):
        for col in range(COLUMNS):
            if board[row][col] == 2:
                encoded[channel_size + row * COLUMNS + col] = 1.0

    # Channel 2: Current player (all 1s if player 1 to move)
    if current_player == 1:
        for i in range(channel_size):
            encoded[2 * channel_size + i] = 1.0

    return encoded


def encode_onehot_3d(board: Board, current_player: Player) -> np.ndarray:
    """
    Same as encode_onehot but returns shape [3, 6, 7] for CNN input.
    """
    flat = encode_onehot(board, current_player)
    return flat.reshape(3, ROWS, COLUMNS)


def encode_flat_binary(board: Board, current_player: Player) -> np.ndarray:
    """
    Encodes a position as a flat binary vector.
    Shape: [85] = [features]

    Features:
    - 42 bits for player 1 pieces
    - 42 bits for player 2 pieces
    - 1 bit for current player

    Simple and efficient for MLP architectures.

    IMPORTANT: This must match encodeFlatBinary() in src/ai/neural/encoding.ts
    """
    cell_count = ROWS * COLUMNS  # 42
    total_size = cell_count * 2 + 1  # 85

    encoded = np.zeros(total_size, dtype=np.float32)

    # First 42: Player 1 pieces
    for row in range(ROWS):
        for col in range(COLUMNS):
            if board[row][col] == 1:
                encoded[row * COLUMNS + col] = 1.0

    # Next 42: Player 2 pieces
    for row in range(ROWS):
        for col in range(COLUMNS):
            if board[row][col] == 2:
                encoded[cell_count + row * COLUMNS + col] = 1.0

    # Last bit: Current player (1 if player 1, 0 if player 2)
    encoded[cell_count * 2] = 1.0 if current_player == 1 else 0.0

    return encoded


def encode_bitboard(board: Board, current_player: Player, move_count: int = 0) -> np.ndarray:
    """
    Encodes a position using bitboard representation.
    Shape: [4] where elements are [p1_bits, p2_bits, player_bit, move_count]

    Each player's position is represented as a bitmask of occupied cells.
    Cell (row, col) maps to bit (row * 7 + col).

    IMPORTANT: This must match encodeBitboard() in src/ai/neural/encoding.ts
    """
    p1_bits = 0
    p2_bits = 0

    for row in range(ROWS):
        for col in range(COLUMNS):
            bit = row * COLUMNS + col
            if board[row][col] == 1:
                p1_bits |= 1 << bit
            elif board[row][col] == 2:
                p2_bits |= 1 << bit

    # Include current player info
    player_bit = 1 if current_player == 1 else 0

    return np.array([p1_bits, p2_bits, player_bit, move_count], dtype=np.float32)


def decode_onehot(encoded: np.ndarray) -> tuple[Board, Player]:
    """
    Decodes a one-hot encoded position back to a board state.
    Useful for debugging and visualization.

    IMPORTANT: This must match decodeOneHot() in src/ai/neural/encoding.ts
    """
    # Handle both flat [126] and shaped [3, 6, 7] input
    if encoded.shape == (3, ROWS, COLUMNS):
        flat = encoded.flatten()
    elif encoded.shape == (126,):
        flat = encoded
    else:
        raise ValueError(f"Invalid one-hot encoding shape: {encoded.shape}")

    channel_size = ROWS * COLUMNS
    board: Board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]

    # Decode player 1 pieces
    for row in range(ROWS):
        for col in range(COLUMNS):
            if flat[row * COLUMNS + col] == 1:
                board[row][col] = 1

    # Decode player 2 pieces
    for row in range(ROWS):
        for col in range(COLUMNS):
            if flat[channel_size + row * COLUMNS + col] == 1:
                board[row][col] = 2

    # Decode current player
    current_player: Player = 1 if flat[2 * channel_size] == 1 else 2

    return board, current_player


def encode_position(
    board: Board,
    current_player: Player,
    encoding: EncodingType,
    move_count: int = 0,
) -> np.ndarray:
    """
    Encodes a position using the specified encoding type.
    """
    if encoding == "onehot-6x7x3":
        return encode_onehot(board, current_player)
    elif encoding == "flat-binary":
        return encode_flat_binary(board, current_player)
    elif encoding == "bitboard":
        return encode_bitboard(board, current_player, move_count)
    else:
        raise ValueError(f"Unknown encoding type: {encoding}")


def get_input_shape(encoding: EncodingType) -> tuple[int, ...]:
    """
    Gets the input shape for a given encoding type.
    """
    if encoding == "onehot-6x7x3":
        return (1, 3, ROWS, COLUMNS)
    elif encoding == "flat-binary":
        return (1, 85)
    elif encoding == "bitboard":
        return (1, 4)
    else:
        raise ValueError(f"Unknown encoding type: {encoding}")


def get_input_size(encoding: EncodingType) -> int:
    """
    Gets the total number of input features for a given encoding type.
    """
    shape = get_input_shape(encoding)
    result = 1
    for dim in shape:
        result *= dim
    return result


def board_from_moves(moves: list[int]) -> tuple[Board, Player]:
    """
    Reconstructs a board state from a sequence of moves.

    Args:
        moves: List of column indices (0-6) representing each move in order.

    Returns:
        Tuple of (board, current_player) after all moves are applied.
    """
    board: Board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
    current_player: Player = 1

    for col in moves:
        # Find the lowest empty row in the column
        for row in range(ROWS - 1, -1, -1):
            if board[row][col] is None:
                board[row][col] = current_player
                break
        # Alternate players
        current_player = 2 if current_player == 1 else 1

    return board, current_player


def board_to_string(board: Board) -> str:
    """
    Converts a board to a human-readable string representation.
    """
    symbols = {None: ".", 1: "X", 2: "O"}
    lines = []
    for row in board:
        lines.append(" ".join(symbols[cell] for cell in row))
    lines.append("0 1 2 3 4 5 6")
    return "\n".join(lines)


def flip_board_horizontal(board: Board) -> Board:
    """
    Flips a board horizontally (mirrors columns).
    This is a valid data augmentation for Connect Four.
    """
    return [list(reversed(row)) for row in board]


def flip_move_horizontal(move: int) -> int:
    """
    Flips a move horizontally (mirrors column index).
    """
    return COLUMNS - 1 - move
