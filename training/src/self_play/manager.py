"""
Self-Play Manager

Manages parallel self-play workers for efficient game generation.
"""

import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from ..data import GameRecord, save_games_jsonl
from .config import SelfPlayConfig
from .worker import ModelProtocol, RandomModel, SelfPlayWorker


def _play_games_worker(
    num_games: int,
    temperature: float,
    temperature_threshold: int,
    add_noise: bool,
    noise_alpha: float,
    noise_epsilon: float,
) -> list[dict]:
    """
    Worker function for parallel game generation.

    Note: Returns dicts instead of GameRecords for pickling compatibility
    across process boundaries.
    """
    worker = SelfPlayWorker(
        model=RandomModel(),  # Each process gets its own random model
        temperature=temperature,
        temperature_threshold=temperature_threshold,
        add_noise=add_noise,
        noise_alpha=noise_alpha,
        noise_epsilon=noise_epsilon,
    )

    games = worker.play_games(num_games)
    return [game.to_dict() for game in games]


class SelfPlayManager:
    """
    Manages parallel self-play workers for efficient data generation.

    Supports both batch generation and continuous generation modes.
    """

    def __init__(
        self,
        model: ModelProtocol | None = None,
        config: SelfPlayConfig | None = None,
    ):
        """
        Initialize the self-play manager.

        Args:
            model: Neural network model for policy prediction.
            config: Self-play configuration.
        """
        self.model = model if model is not None else RandomModel()
        self.config = config if config is not None else SelfPlayConfig()
        self._stop_event = threading.Event()

    def generate_batch(
        self,
        num_games: int | None = None,
        show_progress: bool = True,
    ) -> list[GameRecord]:
        """
        Generate a batch of self-play games.

        Uses multiprocessing for parallel generation when using random policy,
        falls back to threading when using a neural network model.

        Args:
            num_games: Number of games to generate. Defaults to config value.
            show_progress: Whether to show progress bar.

        Returns:
            List of GameRecord objects.
        """
        num_games = num_games or self.config.games_per_iteration
        num_workers = self.config.num_workers

        # Distribute games across workers
        games_per_worker = num_games // num_workers
        remainder = num_games % num_workers

        # For random models, use multiprocessing
        if isinstance(self.model, RandomModel):
            return self._generate_batch_multiprocess(
                num_games, num_workers, games_per_worker, remainder, show_progress
            )
        else:
            # For neural models, use threading to share the model
            return self._generate_batch_threaded(
                num_games, num_workers, games_per_worker, remainder, show_progress
            )

    def _generate_batch_multiprocess(
        self,
        num_games: int,
        num_workers: int,
        games_per_worker: int,
        remainder: int,
        show_progress: bool,
    ) -> list[GameRecord]:
        """Generate games using multiprocessing."""
        games = []
        futures = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(num_workers):
                worker_games = games_per_worker + (1 if i < remainder else 0)
                if worker_games > 0:
                    future = executor.submit(
                        _play_games_worker,
                        worker_games,
                        self.config.temperature,
                        self.config.temperature_threshold,
                        self.config.noise.enabled,
                        self.config.noise.alpha,
                        self.config.noise.epsilon,
                    )
                    futures.append(future)

            # Collect results
            if show_progress:
                pbar = tqdm(total=num_games, desc="Generating games")

            for future in futures:
                game_dicts = future.result()
                for game_dict in game_dicts:
                    games.append(GameRecord.from_dict(game_dict))
                    if show_progress:
                        pbar.update(1)

            if show_progress:
                pbar.close()

        return games

    def _generate_batch_threaded(
        self,
        num_games: int,
        num_workers: int,
        games_per_worker: int,
        remainder: int,
        show_progress: bool,
    ) -> list[GameRecord]:
        """Generate games using threading (for shared model access)."""
        games = []
        games_lock = threading.Lock()

        def worker_task(worker_games: int, pbar=None):
            worker = SelfPlayWorker(
                model=self.model,
                temperature=self.config.temperature,
                temperature_threshold=self.config.temperature_threshold,
                add_noise=self.config.noise.enabled,
                noise_alpha=self.config.noise.alpha,
                noise_epsilon=self.config.noise.epsilon,
            )
            for _ in range(worker_games):
                game = worker.play_game()
                with games_lock:
                    games.append(game)
                    if pbar:
                        pbar.update(1)

        pbar = tqdm(total=num_games, desc="Generating games") if show_progress else None

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                worker_games = games_per_worker + (1 if i < remainder else 0)
                if worker_games > 0:
                    future = executor.submit(worker_task, worker_games, pbar)
                    futures.append(future)

            # Wait for completion
            for future in futures:
                future.result()

        if pbar:
            pbar.close()

        return games

    def generate_continuous(
        self,
        output_queue: queue.Queue,
        callback: Callable[[GameRecord], None] | None = None,
    ) -> None:
        """
        Continuously generate games and put them in a queue.

        Runs until stop() is called.

        Args:
            output_queue: Queue to put generated games into.
            callback: Optional callback function called for each game.
        """
        self._stop_event.clear()

        worker = SelfPlayWorker(
            model=self.model,
            temperature=self.config.temperature,
            temperature_threshold=self.config.temperature_threshold,
            add_noise=self.config.noise.enabled,
            noise_alpha=self.config.noise.alpha,
            noise_epsilon=self.config.noise.epsilon,
        )

        while not self._stop_event.is_set():
            game = worker.play_game()
            output_queue.put(game)
            if callback:
                callback(game)

    def stop(self) -> None:
        """Stop continuous generation."""
        self._stop_event.set()

    def generate_and_save(
        self,
        total_games: int,
        output_dir: str | Path,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[Path]:
        """
        Generate games and save to JSONL files.

        Args:
            total_games: Total number of games to generate.
            output_dir: Directory to save game files.
            batch_size: Number of games per file. Defaults to config.save_interval.
            show_progress: Whether to show progress bar.

        Returns:
            List of paths to saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = batch_size or self.config.save_interval
        saved_files = []
        all_games = []
        file_counter = 0

        pbar = tqdm(total=total_games, desc="Generating games") if show_progress else None

        games_generated = 0
        while games_generated < total_games:
            # Generate a batch
            games_to_generate = min(
                self.config.games_per_iteration,
                total_games - games_generated,
            )
            games = self.generate_batch(games_to_generate, show_progress=False)
            all_games.extend(games)
            games_generated += len(games)

            if pbar:
                pbar.update(len(games))

            # Save if we've accumulated enough games
            while len(all_games) >= batch_size:
                batch = all_games[:batch_size]
                all_games = all_games[batch_size:]

                file_path = output_dir / f"games_{file_counter:06d}.jsonl"
                save_games_jsonl(batch, file_path)
                saved_files.append(file_path)
                file_counter += 1

        # Save remaining games
        if all_games:
            file_path = output_dir / f"games_{file_counter:06d}.jsonl"
            save_games_jsonl(all_games, file_path)
            saved_files.append(file_path)

        if pbar:
            pbar.close()

        return saved_files

    def get_statistics(self, games: list[GameRecord]) -> dict:
        """
        Calculate statistics for a batch of games.

        Args:
            games: List of game records.

        Returns:
            Dictionary of statistics.
        """
        total_positions = sum(len(g.positions) for g in games)
        results = {"player1_win": 0, "player2_win": 0, "draw": 0}
        total_moves = []

        for game in games:
            result = game.metadata.get("result", "unknown")
            if result in results:
                results[result] += 1
            total_moves.append(game.metadata.get("total_moves", len(game.positions)))

        avg_moves = sum(total_moves) / len(total_moves) if total_moves else 0

        return {
            "total_games": len(games),
            "total_positions": total_positions,
            "average_positions_per_game": total_positions / len(games) if games else 0,
            "average_game_length": avg_moves,
            "results": results,
            "player1_win_rate": results["player1_win"] / len(games) if games else 0,
        }
