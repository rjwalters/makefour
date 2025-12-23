"""
Tournament Configuration

Configuration dataclasses for tournament execution.
"""

from dataclasses import dataclass, field
from typing import Literal
import os


@dataclass
class TournamentConfig:
    """Configuration for tournament execution."""

    # Tournament mode
    mode: Literal["round_robin", "swiss"] = "round_robin"

    # Number of games per matchup
    games_per_match: int = 50

    # Number of rounds for swiss tournament
    swiss_rounds: int = 7

    # Number of parallel workers (None = auto-detect)
    parallel_workers: int | None = None

    # Specific bots to include (None = all)
    bot_filter: list[str] | None = None

    # Bots to exclude
    exclude_bots: list[str] | None = None

    # Random seed for reproducibility
    seed: int | None = None

    # Output format
    output_format: Literal["terminal", "json", "csv"] = "terminal"

    # Output file path (None = stdout)
    output_path: str | None = None

    # Verbose output
    verbose: bool = False

    # Maximum search depth for minimax (None = no cap)
    # Use 4-5 for fast tournaments, None for accurate play
    max_depth: int | None = None

    def get_workers(self) -> int:
        """Get number of workers, auto-detecting if not specified."""
        if self.parallel_workers is not None:
            return self.parallel_workers
        # Leave one core free for system
        return max(1, os.cpu_count() - 1)


# Bot persona configurations matching botPersonas.ts
# Maps bot ID to (depth, error_rate, engine_type)
BOT_CONFIGS: dict[str, dict] = {
    # Minimax bots (from botPersonas.ts)
    "rookie": {"depth": 2, "error_rate": 0.35, "engine": "minimax"},
    "rusty": {"depth": 3, "error_rate": 0.25, "engine": "minimax"},
    "blitz": {"depth": 4, "error_rate": 0.18, "engine": "minimax"},
    "nova": {"depth": 4, "error_rate": 0.15, "engine": "minimax"},
    "neuron": {"depth": 5, "error_rate": 0.12, "engine": "minimax"},
    "scholar": {"depth": 6, "error_rate": 0.08, "engine": "minimax"},
    "viper": {"depth": 5, "error_rate": 0.10, "engine": "minimax"},
    "titan": {"depth": 7, "error_rate": 0.04, "engine": "minimax"},
    "sentinel": {"depth": 10, "error_rate": 0.01, "engine": "minimax"},
    # Deep minimax (Oracle)
    "oracle": {"depth": 42, "error_rate": 0.0, "engine": "deep-minimax"},
    # Neural bots
    "neural-intuition": {
        "model": "selfplay-v3.onnx",
        "temperature": 0.5,
        "error_rate": 0.05,
        "engine": "neural",
    },
    "neural-spark": {
        "model": "mlp-tiny-v1.onnx",
        "temperature": 0.8,
        "error_rate": 0.15,
        "engine": "neural",
    },
    "neural-echo": {
        "model": "selfplay-v1.onnx",
        "temperature": 0.3,
        "error_rate": 0.08,
        "engine": "neural",
    },
    # 2swap strategy bots
    "2swap-claimeven": {"error_rate": 0.05, "engine": "claimeven"},
    "2swap-parity": {"error_rate": 0.05, "engine": "parity"},
    "2swap-threats": {"error_rate": 0.05, "engine": "threat-pairs"},
}

# Base ELO ratings - calibrated via 31,500 game tournament (Dec 2025)
BASE_ELOS: dict[str, int] = {
    "random": 0,
    "rookie": 1065,
    "rusty": 1200,
    "blitz": 1275,
    "nova": 1315,
    "neuron": 1370,
    "scholar": 1490,
    "viper": 1425,
    "titan": 1580,
    "sentinel": 1680,
    "oracle": 2200,
    "neural-intuition": 665,
    "neural-spark": 585,
    "neural-echo": 575,
    "2swap-claimeven": 720,
    "2swap-parity": 1095,
    "2swap-threats": 1115,
}

# All bot IDs in order of strength (approximately)
ALL_BOT_IDS = list(BASE_ELOS.keys())
