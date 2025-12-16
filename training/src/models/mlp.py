"""
Multi-Layer Perceptron models for Connect Four.

Simple feed-forward networks that take flat-encoded board positions
and output policy + value predictions.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .base import ConnectFourModel, PolicyValueHead


class TinyMLP(ConnectFourModel):
    """
    Configurable MLP for Connect Four.

    Architecture: Input -> [Hidden layers with ReLU] -> PolicyValueHead

    Variants:
    - micro: 85 -> 32 -> heads (~3k params)
    - tiny: 85 -> 64 -> 32 -> heads (~7k params)
    - small: 85 -> 128 -> 64 -> heads (~15k params)
    """

    def __init__(
        self,
        input_size: int = 85,
        hidden_sizes: list[int] | None = None,
        policy_hidden: int = 32,
        value_hidden: int = 32,
    ):
        """
        Initialize the MLP.

        Args:
            input_size: Number of input features (85 for flat-binary, 126 for one-hot)
            hidden_sizes: List of hidden layer sizes. Default is [64, 32] (tiny variant)
            policy_hidden: Hidden dimension for policy head
            value_hidden: Hidden dimension for value head
        """
        super().__init__(input_size=input_size)

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self._hidden_sizes = hidden_sizes
        self._architecture_name = f"mlp-{len(hidden_sizes)}x{max(hidden_sizes)}"

        # Build backbone layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.backbone = nn.Sequential(*layers)

        # Policy and value heads
        self.heads = PolicyValueHead(
            input_dim=prev_size,
            policy_hidden=policy_hidden,
            value_hidden=value_hidden,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, input_size)

        Returns:
            Tuple of:
                - policy: Logits for each column (batch, 7)
                - value: Position evaluation (batch, 1) in range [-1, 1]
        """
        # Flatten if needed (for compatibility with CNN-style inputs)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        features = self.backbone(x)
        return self.heads(features)

    def get_config(self) -> dict:
        """Returns model configuration for serialization."""
        config = super().get_config()
        config["hidden_sizes"] = self._hidden_sizes
        return config


def create_mlp_micro(input_size: int = 85) -> TinyMLP:
    """
    Create micro MLP variant (~3k params).

    Architecture: input -> 32 -> heads
    """
    return TinyMLP(
        input_size=input_size,
        hidden_sizes=[32],
        policy_hidden=16,
        value_hidden=16,
    )


def create_mlp_tiny(input_size: int = 85) -> TinyMLP:
    """
    Create tiny MLP variant (~7k params).

    Architecture: input -> 64 -> 32 -> heads
    """
    return TinyMLP(
        input_size=input_size,
        hidden_sizes=[64, 32],
        policy_hidden=32,
        value_hidden=32,
    )


def create_mlp_small(input_size: int = 85) -> TinyMLP:
    """
    Create small MLP variant (~15k params).

    Architecture: input -> 128 -> 64 -> heads
    """
    return TinyMLP(
        input_size=input_size,
        hidden_sizes=[128, 64],
        policy_hidden=32,
        value_hidden=32,
    )
