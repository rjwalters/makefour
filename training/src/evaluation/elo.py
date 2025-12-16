"""
ELO Rating Calculation

Implements standard ELO rating calculations from match results.
"""

from dataclasses import dataclass, field
import math
from typing import Tuple

from .arena import MatchResult, TournamentResult


@dataclass
class RatingEstimate:
    """ELO rating estimate with confidence interval."""

    rating: float
    confidence_low: float
    confidence_high: float
    games_played: int

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return (self.confidence_low, self.confidence_high)


class ELOCalculator:
    """
    Calculate ELO ratings from match results.

    Uses standard ELO formula with configurable K-factor.
    """

    def __init__(
        self,
        k_factor: float = 32,
        initial_rating: float = 1200,
    ):
        """
        Initialize ELO calculator.

        Args:
            k_factor: Rating adjustment factor (higher = more volatile)
            initial_rating: Starting rating for new players
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: dict[str, float] = {}
        self.games_played: dict[str, int] = {}

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.

        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B

        Returns:
            Expected score (0-1) for player A
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, match_result: MatchResult) -> None:
        """
        Update ratings based on match result.

        Args:
            match_result: Result of a match between two players
        """
        id1, id2 = match_result.agent1_id, match_result.agent2_id

        r1 = self.ratings.get(id1, self.initial_rating)
        r2 = self.ratings.get(id2, self.initial_rating)

        # Actual scores
        total = match_result.num_games
        s1 = match_result.agent1_score
        s2 = match_result.agent2_score

        # Expected scores
        e1 = self.expected_score(r1, r2)
        e2 = 1 - e1

        # Update ratings (scale by number of games)
        # Using average K-factor per game rather than per match
        self.ratings[id1] = r1 + self.k_factor * total * (s1 - e1) / total
        self.ratings[id2] = r2 + self.k_factor * total * (s2 - e2) / total

        # Track games played
        self.games_played[id1] = self.games_played.get(id1, 0) + total
        self.games_played[id2] = self.games_played.get(id2, 0) + total

    def update_from_tournament(self, tournament: TournamentResult) -> None:
        """
        Update ratings from tournament results.

        Iteratively updates ratings from all matches.

        Args:
            tournament: Tournament result with multiple matches
        """
        for result in tournament.results:
            self.update_ratings(result)

    def calculate_ratings(self, tournament: TournamentResult) -> dict[str, float]:
        """
        Calculate final ratings from tournament results.

        Runs multiple iterations for convergence.

        Args:
            tournament: Tournament result

        Returns:
            Dictionary mapping agent IDs to ratings
        """
        # Reset ratings
        self.ratings = {}
        self.games_played = {}

        # Run multiple iterations for convergence
        for _ in range(10):
            self.update_from_tournament(tournament)

        return self.ratings.copy()

    def set_anchor_rating(self, agent_id: str, rating: float) -> None:
        """
        Set a known rating for an agent to anchor the scale.

        Args:
            agent_id: ID of agent with known rating
            rating: Known rating to set
        """
        self.ratings[agent_id] = rating

    def get_rating(self, agent_id: str) -> float:
        """Get current rating for an agent."""
        return self.ratings.get(agent_id, self.initial_rating)


def estimate_elo_from_matches(
    agent_id: str,
    match_results: list[MatchResult],
    opponent_elos: dict[str, float],
    confidence_level: float = 0.95,
) -> RatingEstimate:
    """
    Estimate ELO rating from match results against known opponents.

    Uses performance rating calculation:
    - Calculate expected score based on opponent ratings
    - Estimate rating that would produce observed score

    Args:
        agent_id: ID of agent to estimate
        match_results: List of match results for this agent
        opponent_elos: Dictionary of known opponent ELO ratings
        confidence_level: Confidence level for interval (default 95%)

    Returns:
        RatingEstimate with estimated rating and confidence interval
    """
    if not match_results:
        return RatingEstimate(
            rating=1200,
            confidence_low=0,
            confidence_high=2400,
            games_played=0,
        )

    total_games = 0
    total_score = 0.0
    weighted_opponent_rating = 0.0

    for result in match_results:
        # Determine which side our agent is on
        if result.agent1_id == agent_id:
            opp_id = result.agent2_id
            score = result.agent1_score
        elif result.agent2_id == agent_id:
            opp_id = result.agent1_id
            score = result.agent2_score
        else:
            continue

        opp_elo = opponent_elos.get(opp_id, 1200)
        n_games = result.num_games

        total_games += n_games
        total_score += score * n_games
        weighted_opponent_rating += opp_elo * n_games

    if total_games == 0:
        return RatingEstimate(
            rating=1200,
            confidence_low=0,
            confidence_high=2400,
            games_played=0,
        )

    # Average score and opponent rating
    avg_score = total_score / total_games
    avg_opp_rating = weighted_opponent_rating / total_games

    # Performance rating calculation
    # Clamp score to avoid division issues
    avg_score = max(0.01, min(0.99, avg_score))

    # Invert expected score formula
    # expected = 1 / (1 + 10^((R_opp - R) / 400))
    # Solving for R:
    # R = R_opp - 400 * log10((1/S) - 1)
    performance_rating = avg_opp_rating - 400 * math.log10((1 / avg_score) - 1)

    # Confidence interval (approximation using binomial)
    # Standard error for win rate
    se = math.sqrt(avg_score * (1 - avg_score) / total_games)

    # z-score for confidence level
    z = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.64

    # Calculate confidence interval bounds
    score_low = max(0.01, avg_score - z * se)
    score_high = min(0.99, avg_score + z * se)

    rating_low = avg_opp_rating - 400 * math.log10((1 / score_low) - 1)
    rating_high = avg_opp_rating - 400 * math.log10((1 / score_high) - 1)

    return RatingEstimate(
        rating=performance_rating,
        confidence_low=min(rating_low, rating_high),
        confidence_high=max(rating_low, rating_high),
        games_played=total_games,
    )


def calibrate_ratings_with_anchors(
    tournament: TournamentResult,
    anchor_ratings: dict[str, float],
    iterations: int = 100,
) -> dict[str, float]:
    """
    Calculate ratings with known anchor points.

    Runs iterative ELO updates while keeping anchor ratings fixed.
    This helps calibrate unknown agents against known benchmarks.

    Args:
        tournament: Tournament result
        anchor_ratings: Known ratings for some agents
        iterations: Number of calibration iterations

    Returns:
        Dictionary mapping agent IDs to calibrated ratings
    """
    calc = ELOCalculator()

    # Initialize with anchor ratings
    for agent_id, rating in anchor_ratings.items():
        calc.ratings[agent_id] = rating

    # Run iterations
    for _ in range(iterations):
        for result in tournament.results:
            id1, id2 = result.agent1_id, result.agent2_id

            # Get current ratings
            r1 = calc.ratings.get(id1, calc.initial_rating)
            r2 = calc.ratings.get(id2, calc.initial_rating)

            # Calculate expected scores
            e1 = calc.expected_score(r1, r2)
            e2 = 1 - e1

            # Actual scores
            s1 = result.agent1_score
            s2 = result.agent2_score

            # Update non-anchored ratings only
            if id1 not in anchor_ratings:
                calc.ratings[id1] = r1 + calc.k_factor * (s1 - e1)
            if id2 not in anchor_ratings:
                calc.ratings[id2] = r2 + calc.k_factor * (s2 - e2)

    return calc.ratings.copy()
