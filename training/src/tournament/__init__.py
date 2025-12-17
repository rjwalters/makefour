"""
Tournament System for Bot ELO Calibration

Run local tournaments between bots to refine ELO ratings.
"""

from .config import TournamentConfig
from .runner import ParallelTournamentRunner, run_tournament
from .db_sync import DatabaseSync

__all__ = [
    "TournamentConfig",
    "ParallelTournamentRunner",
    "run_tournament",
    "DatabaseSync",
]
