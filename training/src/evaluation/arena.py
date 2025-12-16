"""
Arena System for Match and Tournament Management

Manages matches between agents for ELO measurement.
"""

from dataclasses import dataclass, field
from typing import Iterator, Literal
import time

from ..data import Board, Player, COLUMNS, ROWS
from .agents import Agent, get_legal_moves, check_winner, apply_move


@dataclass
class MatchResult:
    """Result of a match between two agents."""

    agent1_id: str
    agent2_id: str
    wins: dict[str, int]  # {agent1_id: count, agent2_id: count, 'draws': count}
    num_games: int
    total_moves: int = 0
    time_seconds: float = 0.0

    @property
    def agent1_wins(self) -> int:
        return self.wins.get(self.agent1_id, 0)

    @property
    def agent2_wins(self) -> int:
        return self.wins.get(self.agent2_id, 0)

    @property
    def draws(self) -> int:
        return self.wins.get("draws", 0)

    @property
    def agent1_score(self) -> float:
        """Score for agent1 (1 for win, 0.5 for draw, 0 for loss)."""
        return (self.agent1_wins + 0.5 * self.draws) / self.num_games

    @property
    def agent2_score(self) -> float:
        """Score for agent2 (1 for win, 0.5 for draw, 0 for loss)."""
        return (self.agent2_wins + 0.5 * self.draws) / self.num_games


@dataclass
class TournamentResult:
    """Result of a tournament between multiple agents."""

    results: list[MatchResult]
    agent_ids: list[str]
    time_seconds: float = 0.0

    def get_match(self, agent1_id: str, agent2_id: str) -> MatchResult | None:
        """Get match result between two specific agents."""
        for result in self.results:
            if (result.agent1_id == agent1_id and result.agent2_id == agent2_id) or (
                result.agent1_id == agent2_id and result.agent2_id == agent1_id
            ):
                return result
        return None

    def get_wins(self, agent_id: str) -> int:
        """Get total wins for an agent across all matches."""
        total = 0
        for result in self.results:
            if result.agent1_id == agent_id:
                total += result.agent1_wins
            elif result.agent2_id == agent_id:
                total += result.agent2_wins
        return total

    def get_losses(self, agent_id: str) -> int:
        """Get total losses for an agent across all matches."""
        total = 0
        for result in self.results:
            if result.agent1_id == agent_id:
                total += result.agent2_wins
            elif result.agent2_id == agent_id:
                total += result.agent1_wins
        return total

    def get_draws(self, agent_id: str) -> int:
        """Get total draws for an agent across all matches."""
        total = 0
        for result in self.results:
            if agent_id in (result.agent1_id, result.agent2_id):
                total += result.draws
        return total

    def get_total_games(self, agent_id: str) -> int:
        """Get total games played by an agent."""
        total = 0
        for result in self.results:
            if agent_id in (result.agent1_id, result.agent2_id):
                total += result.num_games
        return total


@dataclass
class GameState:
    """Current state of a game."""

    board: Board
    current_player: Player
    move_count: int
    move_history: list[int]


class Arena:
    """
    Manages matches between agents for ELO measurement.

    Supports single matches and round-robin tournaments.
    """

    def __init__(self, agents: dict[str, Agent], seed: int | None = None):
        """
        Initialize the arena.

        Args:
            agents: Dictionary mapping agent IDs to Agent instances
            seed: Random seed for reproducibility
        """
        self.agents = agents
        self._seed = seed
        if seed is not None:
            import random

            random.seed(seed)

    def _create_empty_board(self) -> Board:
        """Create an empty board."""
        return [[None for _ in range(COLUMNS)] for _ in range(ROWS)]

    def play_game(
        self,
        agent1: Agent,
        agent2: Agent,
        starting_player: Literal[1, 2] = 1,
    ) -> tuple[Literal["player1_win", "player2_win", "draw"], int]:
        """
        Play a single game between two agents.

        Args:
            agent1: First agent (plays as player 1 when starting)
            agent2: Second agent (plays as player 2 when starting)
            starting_player: Which player moves first (1 or 2)

        Returns:
            Tuple of (result, total_moves)
        """
        board = self._create_empty_board()
        current_player: Player = starting_player
        move_count = 0

        # Reset agents
        agent1.reset()
        agent2.reset()

        while True:
            # Select active agent
            if current_player == 1:
                agent = agent1
            else:
                agent = agent2

            # Get move
            move = agent.get_move(board, current_player)
            move_count += 1

            # Apply move
            board = apply_move(board, move, current_player)

            # Check for terminal state
            winner = check_winner(board)
            if winner is not None:
                if winner == "draw":
                    return "draw", move_count
                elif winner == 1:
                    return "player1_win", move_count
                else:
                    return "player2_win", move_count

            # Switch player
            current_player = 2 if current_player == 1 else 1

    def run_match(
        self,
        agent1_id: str,
        agent2_id: str,
        num_games: int = 100,
        alternate_colors: bool = True,
        verbose: bool = False,
    ) -> MatchResult:
        """
        Run a match between two agents.

        Args:
            agent1_id: ID of first agent
            agent2_id: ID of second agent
            num_games: Number of games to play
            alternate_colors: Whether to alternate starting player
            verbose: Print progress

        Returns:
            MatchResult with win/loss/draw counts
        """
        agent1 = self.agents[agent1_id]
        agent2 = self.agents[agent2_id]

        wins: dict[str, int] = {agent1_id: 0, agent2_id: 0, "draws": 0}
        total_moves = 0
        start_time = time.time()

        for i in range(num_games):
            # Alternate colors if requested
            if alternate_colors and i % 2 == 1:
                # Swap who plays as player 1
                result, moves = self.play_game(agent2, agent1, starting_player=1)
                # Flip result perspective
                if result == "player1_win":
                    result = "player2_win"
                elif result == "player2_win":
                    result = "player1_win"
            else:
                result, moves = self.play_game(agent1, agent2, starting_player=1)

            total_moves += moves

            # Record result
            if result == "player1_win":
                wins[agent1_id] += 1
            elif result == "player2_win":
                wins[agent2_id] += 1
            else:
                wins["draws"] += 1

            if verbose and (i + 1) % 10 == 0:
                print(f"  Games: {i+1}/{num_games}", end="\r")

        time_taken = time.time() - start_time

        if verbose:
            print(f"  Completed {num_games} games in {time_taken:.1f}s")

        return MatchResult(
            agent1_id=agent1_id,
            agent2_id=agent2_id,
            wins=wins,
            num_games=num_games,
            total_moves=total_moves,
            time_seconds=time_taken,
        )

    def run_tournament(
        self,
        agent_ids: list[str] | None = None,
        num_games_per_match: int = 100,
        verbose: bool = False,
    ) -> TournamentResult:
        """
        Run a round-robin tournament between agents.

        Args:
            agent_ids: List of agent IDs to include (default: all)
            num_games_per_match: Games per match
            verbose: Print progress

        Returns:
            TournamentResult with all match results
        """
        if agent_ids is None:
            agent_ids = list(self.agents.keys())

        results: list[MatchResult] = []
        start_time = time.time()

        # Round-robin: each pair plays once
        for i, id1 in enumerate(agent_ids):
            for id2 in agent_ids[i + 1 :]:
                if verbose:
                    print(f"Match: {id1} vs {id2}")

                result = self.run_match(
                    id1, id2, num_games=num_games_per_match, verbose=verbose
                )
                results.append(result)

        time_taken = time.time() - start_time

        return TournamentResult(
            results=results,
            agent_ids=agent_ids,
            time_seconds=time_taken,
        )

    def evaluate_agent(
        self,
        agent_id: str,
        opponent_ids: list[str],
        num_games_per_opponent: int = 100,
        verbose: bool = False,
    ) -> list[MatchResult]:
        """
        Evaluate a single agent against multiple opponents.

        Args:
            agent_id: ID of agent to evaluate
            opponent_ids: List of opponent IDs
            num_games_per_opponent: Games per opponent
            verbose: Print progress

        Returns:
            List of MatchResult for each opponent
        """
        results: list[MatchResult] = []

        for opp_id in opponent_ids:
            if verbose:
                print(f"Evaluating {agent_id} vs {opp_id}...")

            result = self.run_match(
                agent_id, opp_id, num_games=num_games_per_opponent, verbose=verbose
            )
            results.append(result)

        return results


def quick_match(
    agent1: Agent,
    agent2: Agent,
    num_games: int = 100,
    seed: int | None = None,
) -> MatchResult:
    """
    Quick utility to run a match between two agents.

    Args:
        agent1: First agent
        agent2: Second agent
        num_games: Number of games
        seed: Random seed

    Returns:
        MatchResult
    """
    arena = Arena(
        agents={agent1.name: agent1, agent2.name: agent2},
        seed=seed,
    )
    return arena.run_match(agent1.name, agent2.name, num_games=num_games)
