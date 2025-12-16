"""
Self-Play System for Connect Four Neural Network Training

Provides components for generating training data through self-play.
"""

from .worker import SelfPlayWorker
from .manager import SelfPlayManager
from .replay_buffer import ReplayBuffer
from .config import SelfPlayConfig

__all__ = [
    "SelfPlayWorker",
    "SelfPlayManager",
    "ReplayBuffer",
    "SelfPlayConfig",
]
