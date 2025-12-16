"""
Self-Play Configuration

Dataclass-based configuration for self-play parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class NoiseConfig:
    """Configuration for exploration noise."""

    enabled: bool = True
    alpha: float = 0.3  # Dirichlet alpha parameter
    epsilon: float = 0.25  # Noise mixing weight


@dataclass
class OpponentConfig:
    """Configuration for an opponent type."""

    type: Literal["self", "previous_version", "random", "bot"]
    weight: float = 1.0


@dataclass
class ReplayBufferConfig:
    """Configuration for replay buffer."""

    max_size: int = 100000


@dataclass
class SelfPlayConfig:
    """Configuration for self-play data generation."""

    # Worker settings
    num_workers: int = 4
    games_per_iteration: int = 100

    # Temperature settings
    temperature: float = 1.0
    temperature_threshold: int = 15  # Moves after which temp → 0

    # Exploration noise
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # Replay buffer
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)

    # Opponent configuration
    opponents: list[OpponentConfig] = field(default_factory=lambda: [
        OpponentConfig(type="self", weight=0.7),
        OpponentConfig(type="random", weight=0.3),
    ])

    # Output settings
    output_dir: str = "data/self_play"
    save_interval: int = 1000  # Save every N games

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SelfPlayConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        noise_data = data.pop("noise", {})
        noise = NoiseConfig(**noise_data)

        buffer_data = data.pop("replay_buffer", {})
        replay_buffer = ReplayBufferConfig(**buffer_data)

        opponents_data = data.pop("opponents", [])
        opponents = [OpponentConfig(**op) for op in opponents_data]

        return cls(
            noise=noise,
            replay_buffer=replay_buffer,
            opponents=opponents if opponents else None,
            **{k: v for k, v in data.items() if k not in ["noise", "replay_buffer", "opponents"]},
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            "num_workers": self.num_workers,
            "games_per_iteration": self.games_per_iteration,
            "temperature": self.temperature,
            "temperature_threshold": self.temperature_threshold,
            "noise": {
                "enabled": self.noise.enabled,
                "alpha": self.noise.alpha,
                "epsilon": self.noise.epsilon,
            },
            "replay_buffer": {
                "max_size": self.replay_buffer.max_size,
            },
            "opponents": [
                {"type": op.type, "weight": op.weight}
                for op in self.opponents
            ],
            "output_dir": self.output_dir,
            "save_interval": self.save_interval,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
