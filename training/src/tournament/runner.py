"""
Parallel Tournament Runner

Executes tournaments between bots using multiprocessing for speed.
"""

import time
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import Callable
import os

from tqdm import tqdm

from ..evaluation.arena import Arena, MatchResult, TournamentResult
from ..evaluation.agents import Agent, create_all_agents
from ..evaluation.elo import ELOCalculator
from .config import TournamentConfig, BASE_ELOS


@dataclass
class MatchTask:
    """Task definition for parallel match execution."""

    agent1_id: str
    agent2_id: str
    num_games: int
    seed: int | None = None


@dataclass
class TournamentStats:
    """Statistics from a tournament."""

    results: TournamentResult
    elo_ratings: dict[str, float]
    elo_changes: dict[str, float]  # Change from base ELO
    standings: list[tuple[str, float, int, int, int]]  # (id, elo, wins, losses, draws)


# Global agent cache for worker processes
_worker_agents: dict[str, Agent] | None = None
_worker_models_dir: str | None = None
_worker_max_depth: int | None = None


def _init_worker(models_dir: str | None, max_depth: int | None = None) -> None:
    """Initialize worker process with agents."""
    global _worker_agents, _worker_models_dir, _worker_max_depth
    _worker_models_dir = models_dir
    _worker_max_depth = max_depth
    _worker_agents = create_all_agents(models_dir, max_depth=max_depth)


def _execute_match(task: MatchTask) -> MatchResult:
    """Execute a single match in a worker process."""
    global _worker_agents

    if _worker_agents is None:
        _worker_agents = create_all_agents(_worker_models_dir, max_depth=_worker_max_depth)

    # Create arena with the two agents
    agents = {
        task.agent1_id: _worker_agents[task.agent1_id],
        task.agent2_id: _worker_agents[task.agent2_id],
    }
    arena = Arena(agents, seed=task.seed)

    return arena.run_match(
        task.agent1_id,
        task.agent2_id,
        num_games=task.num_games,
        alternate_colors=True,
    )


class ParallelTournamentRunner:
    """
    Runs tournaments between bots using parallel execution.

    Uses multiprocessing to run matches concurrently for speed.
    """

    def __init__(
        self,
        config: TournamentConfig,
        models_dir: str | None = None,
    ):
        """
        Initialize the tournament runner.

        Args:
            config: Tournament configuration
            models_dir: Directory containing ONNX models for neural agents
        """
        self.config = config
        self.models_dir = models_dir
        self.agents = create_all_agents(models_dir, max_depth=config.max_depth)

    def get_bot_ids(self) -> list[str]:
        """Get list of bot IDs to include in tournament."""
        # Start with all available agents
        all_ids = list(self.agents.keys())

        # Apply filter if specified
        if self.config.bot_filter:
            all_ids = [id for id in all_ids if id in self.config.bot_filter]

        # Apply exclusions
        if self.config.exclude_bots:
            all_ids = [id for id in all_ids if id not in self.config.exclude_bots]

        return all_ids

    def run_tournament(self, bot_ids: list[str] | None = None) -> TournamentStats:
        """
        Run a tournament between the specified bots.

        Args:
            bot_ids: List of bot IDs to include (default: use config)

        Returns:
            TournamentStats with results and ELO ratings
        """
        if bot_ids is None:
            bot_ids = self.get_bot_ids()

        if len(bot_ids) < 2:
            raise ValueError("Need at least 2 bots for a tournament")

        # Generate match tasks
        tasks = self._generate_tasks(bot_ids)

        print(f"Running tournament with {len(bot_ids)} bots, {len(tasks)} matches")
        print(f"Games per match: {self.config.games_per_match}")
        print(f"Total games: {len(tasks) * self.config.games_per_match}")
        print(f"Workers: {self.config.get_workers()}")
        print()

        start_time = time.time()

        # Run matches in parallel
        results = self._run_parallel(tasks)

        elapsed = time.time() - start_time

        # Create tournament result
        tournament_result = TournamentResult(
            results=results,
            agent_ids=bot_ids,
            time_seconds=elapsed,
        )

        # Calculate ELO ratings
        elo_ratings = self._calculate_elos(tournament_result)
        elo_changes = {
            id: elo_ratings.get(id, BASE_ELOS.get(id, 1000)) - BASE_ELOS.get(id, 1000)
            for id in bot_ids
        }

        # Generate standings
        standings = self._generate_standings(tournament_result, elo_ratings)

        return TournamentStats(
            results=tournament_result,
            elo_ratings=elo_ratings,
            elo_changes=elo_changes,
            standings=standings,
        )

    def _generate_tasks(self, bot_ids: list[str]) -> list[MatchTask]:
        """Generate match tasks for round-robin tournament."""
        tasks = []
        seed_counter = self.config.seed or 0

        for i, id1 in enumerate(bot_ids):
            for id2 in bot_ids[i + 1 :]:
                tasks.append(
                    MatchTask(
                        agent1_id=id1,
                        agent2_id=id2,
                        num_games=self.config.games_per_match,
                        seed=seed_counter if self.config.seed else None,
                    )
                )
                seed_counter += 1

        return tasks

    def _run_parallel(self, tasks: list[MatchTask]) -> list[MatchResult]:
        """Run matches in parallel using multiprocessing."""
        num_workers = self.config.get_workers()

        # Flush stdout before progress bar
        import sys
        sys.stdout.flush()

        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(self.models_dir, self.config.max_depth),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_execute_match, tasks),
                    total=len(tasks),
                    desc="Matches",
                    unit="match",
                    file=sys.stderr,  # Use stderr for progress
                )
            )

        return results

    def _calculate_elos(self, tournament: TournamentResult) -> dict[str, float]:
        """Calculate ELO ratings from tournament results."""
        # Initialize with base ELOs
        ratings = {id: float(BASE_ELOS.get(id, 1000)) for id in tournament.agent_ids}

        # Use iterative ELO calculation
        calculator = ELOCalculator(k_factor=32)

        # Run multiple iterations for convergence
        for _ in range(10):
            for result in tournament.results:
                id1, id2 = result.agent1_id, result.agent2_id

                # Calculate expected scores
                expected1 = calculator.expected_score(ratings[id1], ratings[id2])
                expected2 = 1 - expected1

                # Actual scores
                actual1 = result.agent1_score
                actual2 = result.agent2_score

                # Update ratings (scaled by number of games)
                scale = result.num_games / 10  # Normalize to ~10 games worth of K
                ratings[id1] += calculator.k_factor * scale * (actual1 - expected1)
                ratings[id2] += calculator.k_factor * scale * (actual2 - expected2)

        return ratings

    def _generate_standings(
        self,
        tournament: TournamentResult,
        elo_ratings: dict[str, float],
    ) -> list[tuple[str, float, int, int, int]]:
        """Generate standings sorted by ELO."""
        standings = []

        for id in tournament.agent_ids:
            wins = tournament.get_wins(id)
            losses = tournament.get_losses(id)
            draws = tournament.get_draws(id)
            elo = elo_ratings.get(id, 1000)
            standings.append((id, elo, wins, losses, draws))

        # Sort by ELO descending
        standings.sort(key=lambda x: x[1], reverse=True)

        return standings


def run_tournament(
    bot_ids: list[str] | None = None,
    games_per_match: int = 50,
    models_dir: str | None = None,
    workers: int | None = None,
    verbose: bool = True,
) -> TournamentStats:
    """
    Convenience function to run a tournament.

    Args:
        bot_ids: List of bot IDs to include (default: all)
        games_per_match: Number of games per matchup
        models_dir: Directory containing ONNX models
        workers: Number of parallel workers
        verbose: Print progress

    Returns:
        TournamentStats with results
    """
    config = TournamentConfig(
        games_per_match=games_per_match,
        parallel_workers=workers,
        bot_filter=bot_ids,
        verbose=verbose,
    )

    runner = ParallelTournamentRunner(config, models_dir)
    return runner.run_tournament()


def format_standings(stats: TournamentStats) -> str:
    """Format tournament standings as a string."""
    lines = []
    lines.append("=" * 70)
    lines.append("TOURNAMENT RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total matches: {len(stats.results.results)}")
    lines.append(
        f"Total games: {sum(r.num_games for r in stats.results.results)}"
    )
    lines.append(f"Time: {stats.results.time_seconds:.1f}s")
    lines.append("")
    lines.append("Final Standings:")
    lines.append("-" * 70)
    lines.append(
        f"{'Rank':<5} {'Bot':<20} {'ELO':>6} {'Change':>8} "
        f"{'W':>5} {'L':>5} {'D':>5} {'Win%':>7}"
    )
    lines.append("-" * 70)

    for rank, (id, elo, wins, losses, draws) in enumerate(stats.standings, 1):
        total = wins + losses + draws
        win_pct = (wins + 0.5 * draws) / total * 100 if total > 0 else 0
        change = stats.elo_changes.get(id, 0)
        change_str = f"+{change:.0f}" if change >= 0 else f"{change:.0f}"

        lines.append(
            f"{rank:<5} {id:<20} {elo:>6.0f} {change_str:>8} "
            f"{wins:>5} {losses:>5} {draws:>5} {win_pct:>6.1f}%"
        )

    lines.append("-" * 70)

    return "\n".join(lines)
