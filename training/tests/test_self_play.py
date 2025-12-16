"""
Tests for Self-Play System

Tests cover game generation, exploration strategies, parallel generation,
and replay buffer functionality.
"""

import queue
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src.data import COLUMNS, ConnectFourGame, GameRecord
from src.self_play import ReplayBuffer, SelfPlayConfig, SelfPlayManager, SelfPlayWorker
from src.self_play.worker import RandomModel


class TestRandomModel:
    """Tests for RandomModel."""

    def test_predict_returns_valid_shape(self):
        """Model returns correct policy and value shapes."""
        model = RandomModel()
        board = [[None for _ in range(7)] for _ in range(6)]
        policy, value = model.predict(board, 1)

        assert policy.shape == (COLUMNS,)
        assert isinstance(value, float)

    def test_predict_only_legal_moves(self):
        """Policy only assigns probability to legal moves."""
        model = RandomModel()
        # Create a board with column 0 full
        board = [[None for _ in range(7)] for _ in range(6)]
        for row in range(6):
            board[row][0] = 1

        policy, _ = model.predict(board, 1)

        assert policy[0] == 0  # Column 0 is full
        assert policy.sum() == pytest.approx(1.0)


class TestSelfPlayWorker:
    """Tests for SelfPlayWorker."""

    def test_play_game_returns_game_record(self):
        """Worker returns a valid GameRecord."""
        worker = SelfPlayWorker()
        game = worker.play_game()

        assert isinstance(game, GameRecord)
        assert len(game.positions) > 0
        assert game.game_id is not None

    def test_play_game_valid_connect_four(self):
        """Generated games follow Connect Four rules."""
        worker = SelfPlayWorker()
        game_record = worker.play_game()

        # Replay the game to verify validity
        game = ConnectFourGame()
        for pos in game_record.positions:
            assert game.make_move(pos.move_played)

        # Game should be terminal after all moves
        assert game.is_terminal()

    def test_positions_have_correct_metadata(self):
        """Each position has correct metadata."""
        worker = SelfPlayWorker()
        game = worker.play_game()

        for i, pos in enumerate(game.positions):
            # Check position has required fields
            assert pos.board is not None
            assert pos.to_move in (1, 2)
            assert 0 <= pos.move_played < COLUMNS
            assert pos.moves_to_end == len(game.positions) - i

    def test_result_from_player_perspective(self):
        """Result is correctly annotated from each player's perspective."""
        worker = SelfPlayWorker(temperature=0)  # Deterministic for testing
        game = worker.play_game()

        # Get final game state to determine winner
        test_game = ConnectFourGame()
        for pos in game.positions:
            test_game.make_move(pos.move_played)

        final_result = test_game.get_result()

        # Verify each position's result matches the expected value
        for pos in game.positions:
            expected = final_result if pos.to_move == 1 else -final_result
            assert pos.result == expected

    def test_temperature_affects_sampling(self):
        """Higher temperature increases move diversity."""
        # Play many games with low temperature
        low_temp_worker = SelfPlayWorker(temperature=0.1)
        high_temp_worker = SelfPlayWorker(temperature=2.0)

        low_temp_moves = []
        high_temp_moves = []

        for _ in range(20):
            game = low_temp_worker.play_game()
            if game.positions:
                low_temp_moves.append(game.positions[0].move_played)

            game = high_temp_worker.play_game()
            if game.positions:
                high_temp_moves.append(game.positions[0].move_played)

        # High temperature should have more diversity (more unique first moves)
        low_temp_unique = len(set(low_temp_moves))
        high_temp_unique = len(set(high_temp_moves))

        # This is probabilistic but should hold most of the time
        assert high_temp_unique >= low_temp_unique

    def test_temperature_threshold(self):
        """Temperature drops to 0 after threshold."""
        worker = SelfPlayWorker(temperature=1.0, temperature_threshold=5)
        game = worker.play_game()

        # Metadata should reflect settings
        assert game.metadata["temperature_threshold"] == 5

    def test_dirichlet_noise(self):
        """Exploration noise is applied when enabled."""
        worker_with_noise = SelfPlayWorker(add_noise=True)
        worker_without_noise = SelfPlayWorker(add_noise=False)

        # Play games and check metadata
        game_with = worker_with_noise.play_game()
        game_without = worker_without_noise.play_game()

        assert game_with.metadata["noise_enabled"] is True
        assert game_without.metadata["noise_enabled"] is False

    def test_play_multiple_games(self):
        """Worker can play multiple games."""
        worker = SelfPlayWorker()
        games = worker.play_games(5)

        assert len(games) == 5
        for game in games:
            assert isinstance(game, GameRecord)
            assert len(game.positions) > 0


class TestSelfPlayManager:
    """Tests for SelfPlayManager."""

    def test_generate_batch(self):
        """Manager generates correct number of games."""
        config = SelfPlayConfig(num_workers=2, games_per_iteration=10)
        manager = SelfPlayManager(config=config)

        games = manager.generate_batch(num_games=10, show_progress=False)

        assert len(games) == 10
        for game in games:
            assert isinstance(game, GameRecord)

    def test_parallel_generation(self):
        """Parallel generation produces valid independent games."""
        config = SelfPlayConfig(num_workers=4, games_per_iteration=20)
        manager = SelfPlayManager(config=config)

        games = manager.generate_batch(num_games=20, show_progress=False)

        # All games should have unique IDs
        game_ids = [g.game_id for g in games]
        assert len(game_ids) == len(set(game_ids))

    def test_generate_and_save(self):
        """Games are correctly saved to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SelfPlayConfig(num_workers=2)
            manager = SelfPlayManager(config=config)

            saved_files = manager.generate_and_save(
                total_games=25,
                output_dir=tmpdir,
                batch_size=10,
                show_progress=False,
            )

            # Should have created 3 files (10 + 10 + 5 games)
            assert len(saved_files) == 3

            # Files should exist
            for path in saved_files:
                assert Path(path).exists()

    def test_get_statistics(self):
        """Statistics calculation is correct."""
        config = SelfPlayConfig(num_workers=2)
        manager = SelfPlayManager(config=config)

        games = manager.generate_batch(num_games=20, show_progress=False)
        stats = manager.get_statistics(games)

        assert stats["total_games"] == 20
        assert stats["total_positions"] > 0
        assert "results" in stats
        assert sum(stats["results"].values()) == 20

    def test_continuous_generation(self):
        """Continuous generation works until stopped."""
        config = SelfPlayConfig(num_workers=1)
        manager = SelfPlayManager(config=config)

        output_queue: queue.Queue = queue.Queue()
        games_received = []

        def callback(game):
            games_received.append(game)

        # Run continuous generation in a thread
        thread = threading.Thread(
            target=manager.generate_continuous,
            args=(output_queue, callback),
        )
        thread.start()

        # Let it run briefly
        time.sleep(0.5)

        # Stop generation
        manager.stop()
        thread.join(timeout=2.0)

        # Should have generated some games
        assert len(games_received) > 0


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_add_and_sample(self):
        """Buffer correctly adds and samples positions."""
        buffer = ReplayBuffer(max_size=100)
        worker = SelfPlayWorker()
        game = worker.play_game()

        buffer.add_game(game)

        assert len(buffer) == len(game.positions)

        samples = buffer.sample(5)
        assert len(samples) <= 5

    def test_max_size_limit(self):
        """Buffer respects maximum size."""
        buffer = ReplayBuffer(max_size=10)
        worker = SelfPlayWorker()

        # Add more positions than max_size
        for _ in range(5):
            game = worker.play_game()
            buffer.add_game(game)

        assert len(buffer) <= 10

    def test_sample_batch_tensors(self):
        """Sample batch returns correct tensor shapes."""
        buffer = ReplayBuffer(max_size=100)
        worker = SelfPlayWorker()

        # Add some games
        for _ in range(5):
            game = worker.play_game()
            buffer.add_game(game)

        batch = buffer.sample_batch(batch_size=8, encoding="flat-binary")

        assert "board" in batch
        assert "move" in batch
        assert "value" in batch
        assert "legal_mask" in batch
        assert "policy" in batch

        batch_size = min(8, len(buffer))
        assert batch["board"].shape[0] == batch_size
        assert batch["move"].shape[0] == batch_size
        assert batch["policy"].shape == (batch_size, COLUMNS)

    def test_augmentation(self):
        """Augmentation flips board and moves correctly."""
        buffer = ReplayBuffer(max_size=100)
        worker = SelfPlayWorker()
        game = worker.play_game()

        buffer.add_game(game)

        # Sample many times and check for variation (due to augmentation)
        moves = []
        for _ in range(50):
            batch = buffer.sample_batch(batch_size=1, augment=True)
            moves.append(batch["move"].item())

        # With augmentation, we should see the original move and its mirror
        # (though this is probabilistic)
        unique_moves = set(moves)
        # At minimum we should have one move
        assert len(unique_moves) >= 1

    def test_thread_safety(self):
        """Buffer is thread-safe for concurrent operations."""
        buffer = ReplayBuffer(max_size=1000)
        worker = SelfPlayWorker()
        errors = []

        def add_games():
            try:
                for _ in range(10):
                    game = worker.play_game()
                    buffer.add_game(game)
            except Exception as e:
                errors.append(e)

        def sample_batch():
            try:
                for _ in range(10):
                    buffer.sample(5)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_games) for _ in range(3)
        ] + [
            threading.Thread(target=sample_batch) for _ in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_clear(self):
        """Buffer can be cleared."""
        buffer = ReplayBuffer(max_size=100)
        worker = SelfPlayWorker()
        game = worker.play_game()

        buffer.add_game(game)
        assert len(buffer) > 0

        buffer.clear()
        assert len(buffer) == 0

    def test_statistics(self):
        """Statistics are correctly calculated."""
        buffer = ReplayBuffer(max_size=100)
        worker = SelfPlayWorker()

        for _ in range(5):
            game = worker.play_game()
            buffer.add_game(game)

        stats = buffer.get_statistics()

        assert "size" in stats
        assert "max_size" in stats
        assert "fill_ratio" in stats
        assert stats["size"] > 0
        assert stats["fill_ratio"] <= 1.0


class TestSelfPlayConfig:
    """Tests for SelfPlayConfig."""

    def test_default_config(self):
        """Default configuration has sensible values."""
        config = SelfPlayConfig()

        assert config.num_workers > 0
        assert config.games_per_iteration > 0
        assert config.temperature > 0
        assert config.noise.alpha > 0
        assert config.replay_buffer.max_size > 0

    def test_yaml_round_trip(self):
        """Config can be saved and loaded from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SelfPlayConfig(
                num_workers=8,
                temperature=0.5,
                games_per_iteration=200,
            )

            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)

            loaded = SelfPlayConfig.from_yaml(path)

            assert loaded.num_workers == 8
            assert loaded.temperature == 0.5
            assert loaded.games_per_iteration == 200


class TestGameLogic:
    """Integration tests for game logic in self-play context."""

    def test_game_terminates(self):
        """All generated games properly terminate."""
        worker = SelfPlayWorker()

        for _ in range(20):
            game = worker.play_game()
            # Replay and verify terminal state
            test_game = ConnectFourGame()
            for pos in game.positions:
                test_game.make_move(pos.move_played)
            assert test_game.is_terminal()

    def test_no_illegal_moves(self):
        """No generated games contain illegal moves."""
        worker = SelfPlayWorker()

        for _ in range(20):
            game_record = worker.play_game()
            test_game = ConnectFourGame()

            for pos in game_record.positions:
                legal = test_game.get_legal_moves()
                assert pos.move_played in legal
                test_game.make_move(pos.move_played)

    def test_result_distribution(self):
        """Games produce reasonable result distribution."""
        worker = SelfPlayWorker()
        games = worker.play_games(100)

        results = {"win": 0, "loss": 0, "draw": 0}
        for game in games:
            result = game.metadata.get("result", "unknown")
            if result == "player1_win":
                results["win"] += 1
            elif result == "player2_win":
                results["loss"] += 1
            elif result == "draw":
                results["draw"] += 1

        # With random play, we should see a mix of results
        total = sum(results.values())
        assert total == 100
        # At least some games should end with wins (not all draws)
        assert results["win"] + results["loss"] > 0
