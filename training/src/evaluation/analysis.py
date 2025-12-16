"""
Result Analysis for Model Evaluation

Provides tools for analyzing match results and generating reports.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

from .arena import MatchResult, TournamentResult
from .elo import RatingEstimate, estimate_elo_from_matches
from .agents import REFERENCE_ELOS


@dataclass
class MatchSummary:
    """Summary of a single match."""

    opponent: str
    opponent_elo: int
    games: int
    wins: int
    losses: int
    draws: int
    score: float
    time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report for a model."""

    model: str
    timestamp: str
    estimated_elo: float
    confidence_interval: tuple[float, float]
    matches: list[MatchSummary]
    total_games: int
    time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "timestamp": self.timestamp,
            "estimated_elo": round(self.estimated_elo),
            "confidence_interval": [
                round(self.confidence_interval[0]),
                round(self.confidence_interval[1]),
            ],
            "matches": [m.to_dict() for m in self.matches],
            "total_games": self.total_games,
            "time_seconds": round(self.time_seconds, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())


def analyze_matches(
    model_name: str,
    match_results: list[MatchResult],
    opponent_elos: dict[str, float] | None = None,
) -> EvaluationReport:
    """
    Analyze match results and generate an evaluation report.

    Args:
        model_name: Name of the model being evaluated
        match_results: List of match results
        opponent_elos: Known ELO ratings for opponents (defaults to REFERENCE_ELOS)

    Returns:
        EvaluationReport with full analysis
    """
    if opponent_elos is None:
        opponent_elos = {k: float(v) for k, v in REFERENCE_ELOS.items()}

    # Build match summaries
    summaries: list[MatchSummary] = []
    total_games = 0
    total_time = 0.0

    for result in match_results:
        # Determine which side is the model
        if result.agent1_id == model_name:
            opp_id = result.agent2_id
            wins = result.agent1_wins
            losses = result.agent2_wins
            score = result.agent1_score
        else:
            opp_id = result.agent1_id
            wins = result.agent2_wins
            losses = result.agent1_wins
            score = result.agent2_score

        summaries.append(
            MatchSummary(
                opponent=opp_id,
                opponent_elo=int(opponent_elos.get(opp_id, 1200)),
                games=result.num_games,
                wins=wins,
                losses=losses,
                draws=result.draws,
                score=round(score, 3),
                time_seconds=round(result.time_seconds, 2),
            )
        )

        total_games += result.num_games
        total_time += result.time_seconds

    # Estimate ELO
    estimate = estimate_elo_from_matches(
        agent_id=model_name,
        match_results=match_results,
        opponent_elos=opponent_elos,
    )

    return EvaluationReport(
        model=model_name,
        timestamp=datetime.now().isoformat(),
        estimated_elo=estimate.rating,
        confidence_interval=estimate.confidence_interval,
        matches=summaries,
        total_games=total_games,
        time_seconds=total_time,
    )


@dataclass
class TournamentSummary:
    """Summary of tournament standings."""

    agent_id: str
    elo: float
    wins: int
    losses: int
    draws: int
    games: int
    win_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_tournament(
    tournament: TournamentResult,
    ratings: dict[str, float],
) -> list[TournamentSummary]:
    """
    Generate tournament standings from results.

    Args:
        tournament: Tournament result
        ratings: Calculated ELO ratings

    Returns:
        List of TournamentSummary sorted by ELO
    """
    summaries: list[TournamentSummary] = []

    for agent_id in tournament.agent_ids:
        wins = tournament.get_wins(agent_id)
        losses = tournament.get_losses(agent_id)
        draws = tournament.get_draws(agent_id)
        games = tournament.get_total_games(agent_id)

        win_rate = (wins + 0.5 * draws) / games if games > 0 else 0.0

        summaries.append(
            TournamentSummary(
                agent_id=agent_id,
                elo=round(ratings.get(agent_id, 1200), 1),
                wins=wins,
                losses=losses,
                draws=draws,
                games=games,
                win_rate=round(win_rate, 3),
            )
        )

    # Sort by ELO descending
    summaries.sort(key=lambda x: x.elo, reverse=True)

    return summaries


def format_tournament_table(summaries: list[TournamentSummary]) -> str:
    """
    Format tournament standings as a text table.

    Args:
        summaries: List of tournament summaries

    Returns:
        Formatted table string
    """
    lines = [
        "╔══════════════════╦═══════╦══════╦══════╦═══════╦═══════╦══════════╗",
        "║      Agent       ║  ELO  ║ Wins ║ Loss ║ Draws ║ Games ║ Win Rate ║",
        "╠══════════════════╬═══════╬══════╬══════╬═══════╬═══════╬══════════╣",
    ]

    for s in summaries:
        lines.append(
            f"║ {s.agent_id:16s} ║ {s.elo:5.0f} ║ {s.wins:4d} ║ {s.losses:4d} ║ {s.draws:5d} ║ {s.games:5d} ║ {s.win_rate:8.1%} ║"
        )

    lines.append("╚══════════════════╩═══════╩══════╩══════╩═══════╩═══════╩══════════╝")

    return "\n".join(lines)


def format_evaluation_report(report: EvaluationReport) -> str:
    """
    Format evaluation report as readable text.

    Args:
        report: Evaluation report

    Returns:
        Formatted text string
    """
    lines = [
        f"═══════════════════════════════════════════════════",
        f"  Model Evaluation Report: {report.model}",
        f"═══════════════════════════════════════════════════",
        f"",
        f"  Estimated ELO: {report.estimated_elo:.0f}",
        f"  95% Confidence: [{report.confidence_interval[0]:.0f}, {report.confidence_interval[1]:.0f}]",
        f"  Total Games: {report.total_games}",
        f"  Time: {report.time_seconds:.1f}s",
        f"",
        f"  Match Results:",
        f"  ─────────────────────────────────────────────────",
    ]

    for m in report.matches:
        lines.append(
            f"  vs {m.opponent:12s} (ELO {m.opponent_elo:4d}): "
            f"{m.wins:3d}W / {m.losses:3d}L / {m.draws:3d}D  "
            f"({m.score:.1%})"
        )

    lines.append(f"  ─────────────────────────────────────────────────")
    lines.append(f"  Generated: {report.timestamp}")

    return "\n".join(lines)


def compare_models(
    reports: list[EvaluationReport],
) -> str:
    """
    Generate a comparison table for multiple models.

    Args:
        reports: List of evaluation reports

    Returns:
        Formatted comparison table
    """
    # Sort by ELO
    sorted_reports = sorted(reports, key=lambda r: r.estimated_elo, reverse=True)

    lines = [
        "╔════════════════════════╦═══════╦═══════════════════════╦═════════╗",
        "║         Model          ║  ELO  ║    95% Confidence     ║  Games  ║",
        "╠════════════════════════╬═══════╬═══════════════════════╬═════════╣",
    ]

    for r in sorted_reports:
        conf_str = f"[{r.confidence_interval[0]:.0f}, {r.confidence_interval[1]:.0f}]"
        lines.append(
            f"║ {r.model:22s} ║ {r.estimated_elo:5.0f} ║ {conf_str:21s} ║ {r.total_games:7d} ║"
        )

    lines.append("╚════════════════════════╩═══════╩═══════════════════════╩═════════╝")

    return "\n".join(lines)
