"""Data loading and encoding for Connect Four neural network training."""

from .encoding import (
    ROWS,
    COLUMNS,
    Board,
    Player,
    EncodingType,
    encode_onehot,
    encode_onehot_3d,
    encode_flat_binary,
    encode_bitboard,
    decode_onehot,
    encode_position,
    get_input_shape,
    get_input_size,
    board_from_moves,
    board_to_string,
    flip_board_horizontal,
    flip_move_horizontal,
)
from .dataset import (
    ConnectFourDataset,
    GameRecord,
    Position,
    save_games_jsonl,
    load_games_jsonl,
    create_train_val_test_split,
)
from .game import ConnectFourGame, get_legal_moves, check_winner, make_move
from .export import (
    convert_api_export,
    generate_synthetic_dataset,
    generate_biased_dataset,
    load_api_export_file,
    get_dataset_statistics,
    filter_games,
)
from .validation import (
    validate_board,
    validate_move,
    validate_position,
    validate_game_record,
    validate_dataset,
    find_duplicate_positions,
    ValidationResult,
)

__all__ = [
    # Constants and types
    "ROWS",
    "COLUMNS",
    "Board",
    "Player",
    "EncodingType",
    # Encoding functions
    "encode_onehot",
    "encode_onehot_3d",
    "encode_flat_binary",
    "encode_bitboard",
    "decode_onehot",
    "encode_position",
    "get_input_shape",
    "get_input_size",
    "board_from_moves",
    "board_to_string",
    "flip_board_horizontal",
    "flip_move_horizontal",
    # Dataset classes
    "ConnectFourDataset",
    "GameRecord",
    "Position",
    "save_games_jsonl",
    "load_games_jsonl",
    "create_train_val_test_split",
    # Game logic
    "ConnectFourGame",
    "get_legal_moves",
    "check_winner",
    "make_move",
    # Export utilities
    "convert_api_export",
    "generate_synthetic_dataset",
    "generate_biased_dataset",
    "load_api_export_file",
    "get_dataset_statistics",
    "filter_games",
    # Validation
    "validate_board",
    "validate_move",
    "validate_position",
    "validate_game_record",
    "validate_dataset",
    "find_duplicate_positions",
    "ValidationResult",
]
