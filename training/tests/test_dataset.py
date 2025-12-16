"""Tests for dataset classes and utilities."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import (
    Position,
    GameRecord,
    ConnectFourDataset,
    save_games_jsonl,
    load_games_jsonl,
    create_train_val_test_split,
)
from data.encoding import ROWS, COLUMNS


class TestPosition:
    """Test Position dataclass."""

    def test_create_position(self):
        board = [[None] * COLUMNS for _ in range(ROWS)]
        pos = Position(
            board=board,
            to_move=1,
            move_played=3,
            result=1.0,
            moves_to_end=5,
        )

        assert pos.to_move == 1
        assert pos.move_played == 3
        assert pos.result == 1.0
        assert pos.moves_to_end == 5
        assert pos.policy_target is None
        assert pos.value_target is None


class TestGameRecord:
    """Test GameRecord dataclass."""

    def test_from_moves_simple(self):
        """Create game record from simple moves."""
        record = GameRecord.from_moves("test-1", [3, 4, 3])

        assert record.game_id == "test-1"
        assert len(record.positions) == 3

        # First position is empty board, P1 to move
        assert record.positions[0].to_move == 1
        assert record.positions[0].move_played == 3
        assert record.positions[0].moves_to_end == 3

    def test_from_moves_win(self):
        """Create game record from winning game."""
        # P1 vertical win in column 3
        moves = [3, 4, 3, 4, 3, 4, 3]
        record = GameRecord.from_moves("win-game", moves)

        assert len(record.positions) == 7

        # First position should have result from P1's perspective
        # P1 wins, so result = 1.0
        assert record.positions[0].result == 1.0

        # Second position is from P2's perspective (they lose)
        assert record.positions[1].result == -1.0

    def test_to_dict_from_dict(self):
        """Round-trip through dictionary."""
        record = GameRecord.from_moves("test-2", [3, 4, 3, 4])

        # Convert to dict
        data = record.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        # Convert back
        restored = GameRecord.from_dict(restored_data)

        assert restored.game_id == record.game_id
        assert len(restored.positions) == len(record.positions)

    def test_metadata(self):
        """Metadata is preserved."""
        metadata = {"source": "test", "elo": 1200}
        record = GameRecord.from_moves("test-3", [3], metadata)

        assert record.metadata["source"] == "test"
        assert record.metadata["elo"] == 1200


class TestConnectFourDataset:
    """Test PyTorch dataset."""

    def test_create_from_games(self):
        """Create dataset from game records."""
        games = [
            GameRecord.from_moves("g1", [3, 4, 3]),
            GameRecord.from_moves("g2", [2, 3, 2, 3]),
        ]

        dataset = ConnectFourDataset(games=games, augment=False)

        # 3 + 4 = 7 positions
        assert len(dataset) == 7

    def test_augmentation_doubles_size(self):
        """Augmentation doubles dataset size."""
        games = [GameRecord.from_moves("g1", [3, 4, 3])]

        without_aug = ConnectFourDataset(games=games, augment=False)
        with_aug = ConnectFourDataset(games=games, augment=True)

        assert len(with_aug) == 2 * len(without_aug)

    def test_getitem_returns_dict(self):
        """__getitem__ returns expected dictionary."""
        games = [GameRecord.from_moves("g1", [3, 4])]
        dataset = ConnectFourDataset(games=games, augment=False, encoding="flat-binary")

        item = dataset[0]

        assert "board" in item
        assert "move" in item
        assert "value" in item
        assert "legal_mask" in item
        assert "to_move" in item
        assert "moves_to_end" in item

        # Check shapes
        assert item["board"].shape == (85,)
        assert item["legal_mask"].shape == (7,)

    def test_different_encodings(self):
        """Test different encoding types."""
        games = [GameRecord.from_moves("g1", [3, 4])]

        flat_ds = ConnectFourDataset(games=games, augment=False, encoding="flat-binary")
        onehot_ds = ConnectFourDataset(games=games, augment=False, encoding="onehot-6x7x3")

        assert flat_ds[0]["board"].shape == (85,)
        assert onehot_ds[0]["board"].shape == (126,)

    def test_statistics(self):
        """Test statistics computation."""
        games = [
            GameRecord.from_moves("g1", [3, 4, 3, 4, 3, 4, 3]),  # P1 wins
            GameRecord.from_moves("g2", [4, 3, 4, 3, 4, 3, 4]),  # P2 wins
        ]
        dataset = ConnectFourDataset(games=games, augment=False)

        stats = dataset.get_statistics()

        assert stats["total_positions"] == 14
        assert stats["encoding"] == "flat-binary"


class TestJsonlIO:
    """Test JSONL file operations."""

    def test_save_and_load(self):
        """Save and load games from JSONL."""
        games = [
            GameRecord.from_moves("g1", [3, 4, 3]),
            GameRecord.from_moves("g2", [2, 3, 2, 3]),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            save_games_jsonl(games, path)
            loaded = load_games_jsonl(path)

            assert len(loaded) == 2
            assert loaded[0].game_id == "g1"
            assert loaded[1].game_id == "g2"
        finally:
            Path(path).unlink()

    def test_load_from_path(self):
        """Dataset can load directly from path."""
        games = [GameRecord.from_moves("g1", [3, 4, 3])]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            save_games_jsonl(games, path)
            dataset = ConnectFourDataset(data_path=path, augment=False)

            assert len(dataset) == 3
        finally:
            Path(path).unlink()


class TestTrainValTestSplit:
    """Test dataset splitting."""

    def test_default_split(self):
        """Default 80/10/10 split."""
        games = [GameRecord.from_moves(f"g{i}", [3]) for i in range(100)]

        train, val, test = create_train_val_test_split(games)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_custom_split(self):
        """Custom split ratios."""
        games = [GameRecord.from_moves(f"g{i}", [3]) for i in range(100)]

        train, val, test = create_train_val_test_split(
            games, train_ratio=0.7, val_ratio=0.2
        )

        assert len(train) == 70
        assert len(val) == 20
        assert len(test) == 10

    def test_reproducible_split(self):
        """Same seed produces same split."""
        games = [GameRecord.from_moves(f"g{i}", [3]) for i in range(100)]

        train1, val1, test1 = create_train_val_test_split(games, seed=42)
        train2, val2, test2 = create_train_val_test_split(games, seed=42)

        assert [g.game_id for g in train1] == [g.game_id for g in train2]
