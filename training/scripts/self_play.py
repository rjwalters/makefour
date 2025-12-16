#!/usr/bin/env python3
"""
Self-Play Data Generation Script

Generates training data through self-play for neural network training.

Usage:
    python scripts/self_play.py --games 10000 --output data/self_play/
    python scripts/self_play.py --config configs/self_play.yaml --games 10000
"""

import argparse
import sys
import time
from pathlib import Path

# Add training directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.self_play import SelfPlayConfig, SelfPlayManager


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--games",
        type=int,
        required=True,
        help="Number of games to generate",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default="data/self_play",
        help="Output directory for generated games",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of games per output file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--temperature-threshold",
        type=int,
        default=15,
        help="Move count after which temperature drops to 0",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable exploration noise",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = SelfPlayConfig.from_yaml(args.config)
    else:
        config = SelfPlayConfig()

    # Override with command-line arguments
    config.num_workers = args.workers
    config.temperature = args.temperature
    config.temperature_threshold = args.temperature_threshold
    config.noise.enabled = not args.no_noise
    config.output_dir = args.output
    config.save_interval = args.batch_size

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    if not args.quiet:
        print("Self-Play Configuration:")
        print(f"  Games to generate: {args.games}")
        print(f"  Output directory: {output_dir}")
        print(f"  Workers: {config.num_workers}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Temperature threshold: {config.temperature_threshold}")
        print(f"  Noise enabled: {config.noise.enabled}")
        print(f"  Batch size: {args.batch_size}")
        print()

    # Create manager and generate games
    manager = SelfPlayManager(config=config)

    start_time = time.time()
    saved_files = manager.generate_and_save(
        total_games=args.games,
        output_dir=output_dir,
        batch_size=args.batch_size,
        show_progress=not args.quiet,
    )
    elapsed = time.time() - start_time

    # Print summary
    if not args.quiet:
        print()
        print("Generation Complete!")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Games/second: {args.games / elapsed:.1f}")
        print(f"  Files saved: {len(saved_files)}")
        for path in saved_files:
            print(f"    - {path}")


if __name__ == "__main__":
    main()
