"""
Model Evaluation Harness

Provides tools for measuring model strength (ELO) against known opponents.
"""

from .agents import (
    Agent,
    RandomAgent,
    MinimaxAgent,
    NeuralAgent,
    REFERENCE_AGENTS,
    REFERENCE_ELOS,
)
from .arena import Arena, MatchResult, TournamentResult, quick_match
from .elo import ELOCalculator, estimate_elo_from_matches, RatingEstimate
from .analysis import (
    analyze_matches,
    analyze_tournament,
    EvaluationReport,
    TournamentSummary,
    format_evaluation_report,
    format_tournament_table,
    compare_models,
)

__all__ = [
    # Agents
    "Agent",
    "RandomAgent",
    "MinimaxAgent",
    "NeuralAgent",
    "REFERENCE_AGENTS",
    "REFERENCE_ELOS",
    # Arena
    "Arena",
    "MatchResult",
    "TournamentResult",
    "quick_match",
    # ELO
    "ELOCalculator",
    "estimate_elo_from_matches",
    "RatingEstimate",
    # Analysis
    "analyze_matches",
    "analyze_tournament",
    "EvaluationReport",
    "TournamentSummary",
    "format_evaluation_report",
    "format_tournament_table",
    "compare_models",
]
