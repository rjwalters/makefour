#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluate neural network models against reference opponents to measure ELO.

Usage:
    # Evaluate single model against reference pool
    python scripts/evaluate.py \
        --model models/cnn-tiny-v1.onnx \
        --opponents rookie,nova,scholar,titan \
        --games-per-opponent 100 \
        --output results/cnn-tiny-v1-eval.json

    # Run full tournament
    python scripts/evaluate.py \
        --tournament \
        --models models/*.onnx \
        --include-reference \
        --games-per-match 50 \
        --output results/tournament.json

    # Quick sanity check (10 games vs random, rookie, nova)
    python scripts/evaluate.py \
        --model models/cnn-tiny-v1.onnx \
        --quick
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import (
    Arena,
    NeuralAgent,
    REFERENCE_AGENTS,
    REFERENCE_ELOS,
    analyze_matches,
    analyze_tournament,
    format_evaluation_report,
    format_tournament_table,
    ELOCalculator,
    calibrate_ratings_with_anchors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Connect Four models against reference opponents"
    )

    # Model specification
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        help="Path to single ONNX model file to evaluate",
    )
    model_group.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Paths to multiple ONNX model files (supports glob patterns)",
    )

    # Evaluation mode
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Run round-robin tournament instead of evaluation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick sanity check (10 games vs random, rookie, nova)",
    )

    # Opponents
    parser.add_argument(
        "--opponents",
        type=str,
        default="rookie,nova,scholar,titan",
        help="Comma-separated list of reference opponents",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Include reference agents in tournament",
    )

    # Game configuration
    parser.add_argument(
        "--games-per-opponent",
        type=int,
        default=100,
        help="Number of games per opponent in evaluation mode",
    )
    parser.add_argument(
        "--games-per-match",
        type=int,
        default=100,
        help="Number of games per match in tournament mode",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def expand_model_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns to actual file paths."""
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
        else:
            print(f"Warning: No files found for pattern: {pattern}")
    return sorted(set(paths))


def run_quick_evaluation(model_path: str, verbose: bool = False) -> None:
    """Run quick sanity check evaluation."""
    print(f"Quick evaluation: {model_path}")
    print()

    # Create neural agent
    try:
        neural_agent = NeuralAgent(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Quick opponents
    opponents = ["random", "rookie", "nova"]
    games_per = 10

    # Build arena
    agents = {neural_agent.name: neural_agent}
    for opp_id in opponents:
        if opp_id in REFERENCE_AGENTS:
            agents[opp_id] = REFERENCE_AGENTS[opp_id]

    arena = Arena(agents)

    print(f"Running {games_per} games against each: {', '.join(opponents)}")
    print()

    for opp_id in opponents:
        result = arena.run_match(
            neural_agent.name, opp_id, num_games=games_per, verbose=verbose
        )
        wins = result.wins.get(neural_agent.name, 0)
        losses = result.wins.get(opp_id, 0)
        draws = result.draws
        print(f"  vs {opp_id:8s}: {wins}W / {losses}L / {draws}D")

    print()
    print("Quick check complete!")


def run_evaluation(
    model_path: str,
    opponent_ids: list[str],
    games_per_opponent: int,
    output_path: str | None,
    verbose: bool,
    seed: int | None,
) -> None:
    """Run full model evaluation."""
    print(f"Evaluating model: {model_path}")
    print(f"Opponents: {', '.join(opponent_ids)}")
    print(f"Games per opponent: {games_per_opponent}")
    print()

    # Create neural agent
    try:
        neural_agent = NeuralAgent(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Build arena
    agents = {neural_agent.name: neural_agent}
    for opp_id in opponent_ids:
        if opp_id in REFERENCE_AGENTS:
            agents[opp_id] = REFERENCE_AGENTS[opp_id]
        else:
            print(f"Warning: Unknown opponent '{opp_id}', skipping")

    arena = Arena(agents, seed=seed)

    # Run evaluation
    start_time = time.time()
    match_results = arena.evaluate_agent(
        neural_agent.name,
        [opp for opp in opponent_ids if opp in agents],
        num_games_per_opponent=games_per_opponent,
        verbose=verbose,
    )
    total_time = time.time() - start_time

    # Analyze results
    report = analyze_matches(
        neural_agent.name,
        match_results,
        opponent_elos={k: float(v) for k, v in REFERENCE_ELOS.items()},
    )

    # Print report
    print()
    print(format_evaluation_report(report))

    # Save if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save(output_path)
        print()
        print(f"Results saved to: {output_path}")


def run_tournament(
    model_paths: list[str],
    include_reference: bool,
    games_per_match: int,
    output_path: str | None,
    verbose: bool,
    seed: int | None,
) -> None:
    """Run round-robin tournament."""
    print(f"Tournament with {len(model_paths)} models")
    if include_reference:
        print(f"Including reference agents: {list(REFERENCE_AGENTS.keys())}")
    print(f"Games per match: {games_per_match}")
    print()

    # Build agents dict
    agents: dict = {}

    # Add neural models
    for path in model_paths:
        try:
            agent = NeuralAgent(path)
            agents[agent.name] = agent
            print(f"  Loaded: {agent.name}")
        except Exception as e:
            print(f"  Error loading {path}: {e}")

    # Add reference agents if requested
    if include_reference:
        for agent_id, agent in REFERENCE_AGENTS.items():
            agents[agent_id] = agent

    if len(agents) < 2:
        print("Error: Need at least 2 agents for tournament")
        sys.exit(1)

    # Run tournament
    print()
    print(f"Starting tournament with {len(agents)} agents...")
    arena = Arena(agents, seed=seed)
    tournament = arena.run_tournament(
        games_per_match=games_per_match,
        verbose=verbose,
    )

    # Calculate ELO ratings
    anchor_ratings = {k: float(v) for k, v in REFERENCE_ELOS.items() if k in agents}
    ratings = calibrate_ratings_with_anchors(tournament, anchor_ratings)

    # Analyze and display
    summaries = analyze_tournament(tournament, ratings)
    print()
    print(format_tournament_table(summaries))

    # Save if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result = {
            "type": "tournament",
            "agents": list(agents.keys()),
            "games_per_match": games_per_match,
            "time_seconds": round(tournament.time_seconds, 2),
            "standings": [s.to_dict() for s in summaries],
            "matches": [
                {
                    "agent1": r.agent1_id,
                    "agent2": r.agent2_id,
                    "agent1_wins": r.agent1_wins,
                    "agent2_wins": r.agent2_wins,
                    "draws": r.draws,
                    "games": r.num_games,
                }
                for r in tournament.results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print()
        print(f"Results saved to: {output_path}")


def main() -> None:
    args = parse_args()

    # Quick mode
    if args.quick:
        if not args.model:
            print("Error: --quick requires --model")
            sys.exit(1)
        run_quick_evaluation(args.model, args.verbose)
        return

    # Tournament mode
    if args.tournament:
        if not args.models:
            print("Error: --tournament requires --models")
            sys.exit(1)
        model_paths = expand_model_paths(args.models)
        if not model_paths and not args.include_reference:
            print("Error: No model files found and --include-reference not set")
            sys.exit(1)
        run_tournament(
            model_paths,
            args.include_reference,
            args.games_per_match,
            args.output,
            args.verbose,
            args.seed,
        )
        return

    # Evaluation mode
    if args.model:
        opponent_ids = [x.strip() for x in args.opponents.split(",")]
        run_evaluation(
            args.model,
            opponent_ids,
            args.games_per_opponent,
            args.output,
            args.verbose,
            args.seed,
        )
        return

    # No mode specified
    print("Error: Specify --model, --models with --tournament, or --quick")
    sys.exit(1)


if __name__ == "__main__":
    main()
