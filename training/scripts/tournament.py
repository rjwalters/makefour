#!/usr/bin/env python3
"""
Local Bot Tournament System

Run tournaments between MakeFour bots to calibrate ELO ratings.

Usage:
    # Quick tournament with default settings
    python scripts/tournament.py

    # Full round-robin with all bots
    python scripts/tournament.py --all-bots --games 100

    # Tournament with specific bots
    python scripts/tournament.py --bots rookie,nova,scholar,titan --games 50

    # Exclude slow bots (like oracle)
    python scripts/tournament.py --exclude oracle,sentinel

    # Output to JSON
    python scripts/tournament.py --output results/tournament.json

    # Sync to database (dry-run first)
    python scripts/tournament.py --sync-db --dry-run
    python scripts/tournament.py --sync-db --env remote --no-dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tournament import TournamentConfig, ParallelTournamentRunner, DatabaseSync
from src.tournament.runner import format_standings
from src.tournament.config import BASE_ELOS, ALL_BOT_IDS


def main():
    parser = argparse.ArgumentParser(
        description="Run local bot tournaments to calibrate ELO ratings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 4 bots
  python scripts/tournament.py --bots rookie,blitz,nova,scholar -g 20

  # Full tournament (excluding slow bots)
  python scripts/tournament.py --exclude oracle --games 50

  # Sync results to local database
  python scripts/tournament.py --sync-db --dry-run
  python scripts/tournament.py --sync-db --env local --no-dry-run
        """,
    )

    # Bot selection
    parser.add_argument(
        "--bots",
        "-b",
        type=str,
        default=None,
        help="Comma-separated list of bot IDs to include",
    )
    parser.add_argument(
        "--all-bots",
        action="store_true",
        help="Include all available bots",
    )
    parser.add_argument(
        "--exclude",
        "-x",
        type=str,
        default=None,
        help="Comma-separated list of bot IDs to exclude",
    )
    parser.add_argument(
        "--list-bots",
        action="store_true",
        help="List all available bots and exit",
    )

    # Tournament settings
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=50,
        help="Number of games per matchup (default: 50)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (JSON format)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # Database sync
    parser.add_argument(
        "--sync-db",
        action="store_true",
        help="Sync results to database",
    )
    parser.add_argument(
        "--env",
        choices=["local", "remote"],
        default="local",
        help="Database environment (default: local)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be updated without making changes (default)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually apply updates to database",
    )

    # Models
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing ONNX models (default: training/models)",
    )

    args = parser.parse_args()

    # Handle --list-bots
    if args.list_bots:
        print("Available bots:")
        print("-" * 50)
        print(f"{'ID':<20} {'Base ELO':>10}")
        print("-" * 50)
        for bot_id in sorted(ALL_BOT_IDS, key=lambda x: BASE_ELOS.get(x, 0)):
            elo = BASE_ELOS.get(bot_id, 1000)
            print(f"{bot_id:<20} {elo:>10}")
        return 0

    # Determine models directory
    models_dir = args.models_dir
    if models_dir is None:
        # Default to training/models relative to this script
        models_dir = str(Path(__file__).parent.parent / "models")

    if not os.path.exists(models_dir):
        print(f"Warning: Models directory not found: {models_dir}")
        print("Neural bots will not be available")
        models_dir = None

    # Determine bot IDs
    if args.bots:
        bot_ids = [b.strip() for b in args.bots.split(",")]
    elif args.all_bots:
        bot_ids = list(ALL_BOT_IDS)
    else:
        # Default: exclude oracle (too slow) and random (too weak)
        bot_ids = [b for b in ALL_BOT_IDS if b not in ["oracle", "random"]]

    # Apply exclusions
    if args.exclude:
        exclude_ids = [b.strip() for b in args.exclude.split(",")]
        bot_ids = [b for b in bot_ids if b not in exclude_ids]

    # Validate bot IDs
    available = set(ALL_BOT_IDS)
    invalid = [b for b in bot_ids if b not in available]
    if invalid:
        print(f"Error: Unknown bot IDs: {', '.join(invalid)}")
        print(f"Use --list-bots to see available bots")
        return 1

    if len(bot_ids) < 2:
        print("Error: Need at least 2 bots for a tournament")
        return 1

    # Create configuration
    config = TournamentConfig(
        games_per_match=args.games,
        parallel_workers=args.workers,
        bot_filter=bot_ids,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Run tournament
    print(f"Starting tournament with {len(bot_ids)} bots...")
    print(f"Bots: {', '.join(sorted(bot_ids))}")
    print()

    runner = ParallelTournamentRunner(config, models_dir)
    stats = runner.run_tournament(bot_ids)

    # Print results
    print()
    print(format_standings(stats))

    # Output to file if requested
    if args.output:
        output_data = {
            "config": {
                "games_per_match": config.games_per_match,
                "bots": bot_ids,
                "seed": config.seed,
            },
            "time_seconds": stats.results.time_seconds,
            "standings": [
                {
                    "rank": i + 1,
                    "id": id,
                    "elo": elo,
                    "elo_change": stats.elo_changes.get(id, 0),
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                }
                for i, (id, elo, wins, losses, draws) in enumerate(stats.standings)
            ],
            "matches": [
                {
                    "agent1": r.agent1_id,
                    "agent2": r.agent2_id,
                    "agent1_wins": r.agent1_wins,
                    "agent2_wins": r.agent2_wins,
                    "draws": r.draws,
                }
                for r in stats.results.results
            ],
        }

        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")

    # Database sync if requested
    if args.sync_db:
        print()
        print("=" * 60)
        print("DATABASE SYNC")
        print("=" * 60)

        # Find project root (contains wrangler.toml)
        project_root = Path(__file__).parent.parent.parent
        if not (project_root / "wrangler.toml").exists():
            print("Warning: Could not find wrangler.toml")
            project_root = None

        sync = DatabaseSync(
            env=args.env,
            project_root=str(project_root) if project_root else None,
        )

        # Prepare updates
        updates = sync.prepare_updates(stats.elo_ratings, stats.results)

        if not updates:
            print("No updates to apply")
            return 0

        # Determine dry-run mode
        dry_run = not args.no_dry_run

        # Apply updates
        sync.apply_updates(updates, dry_run=dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
