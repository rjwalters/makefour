"""
Convolutional Neural Network models for Connect Four.

CNNs that process 2D board representations with spatial convolutions.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .base import ConnectFourModel, PolicyValueHead


class TinyCNN(ConnectFourModel):
    """
    Configurable CNN for Connect Four.

    Architecture: Conv layers -> Flatten -> PolicyValueHead

    Input shape: (batch, 3, 6, 7) - 3 channels for one-hot encoding
    - Channel 0: Player 1 pieces
    - Channel 1: Player 2 pieces
    - Channel 2: Current player indicator

    Variants:
    - micro: 2 conv layers, 16 filters (~5k params)
    - tiny: 3 conv layers, 32 filters (~15k params)
    - small: 4 conv layers, 64 filters (~50k params)
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        policy_hidden: int = 32,
        value_hidden: int = 32,
    ):
        """
        Initialize the CNN.

        Args:
            in_channels: Number of input channels (3 for one-hot encoding)
            channels: List of output channels for each conv layer. Default: [32, 32, 32]
            kernel_sizes: List of kernel sizes for each conv layer. Default: [3, 3, 3]
            policy_hidden: Hidden dimension for policy head
            value_hidden: Hidden dimension for value head
        """
        # Input is (3, 6, 7) = 126 features when flattened
        super().__init__(input_size=126, input_shape=(in_channels, 6, 7))

        if channels is None:
            channels = [32, 32, 32]
        if kernel_sizes is None:
            kernel_sizes = [3] * len(channels)

        assert len(channels) == len(kernel_sizes), "channels and kernel_sizes must have same length"

        self._channels = channels
        self._kernel_sizes = kernel_sizes
        self._architecture_name = f"cnn-{len(channels)}x{max(channels)}"

        # Build convolutional layers
        conv_layers = []
        prev_channels = in_channels
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            # Use padding to preserve spatial dimensions where possible
            padding = kernel_size // 2
            conv_layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size, padding=padding)
            )
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            prev_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Calculate flattened size after convolutions
        # With same padding, spatial dims are preserved: 6 x 7
        self._flat_size = prev_channels * 6 * 7

        # Policy and value heads
        self.heads = PolicyValueHead(
            input_dim=self._flat_size,
            policy_hidden=policy_hidden,
            value_hidden=value_hidden,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 3, 6, 7) or (batch, 126)

        Returns:
            Tuple of:
                - policy: Logits for each column (batch, 7)
                - value: Position evaluation (batch, 1) in range [-1, 1]
        """
        # Reshape flat input to 2D if needed
        if x.dim() == 2:
            x = x.view(-1, 3, 6, 7)

        # Apply convolutions
        features = self.conv(x)

        # Flatten for fully connected heads
        features = features.view(features.size(0), -1)

        return self.heads(features)

    def get_config(self) -> dict:
        """Returns model configuration for serialization."""
        config = super().get_config()
        config["channels"] = self._channels
        config["kernel_sizes"] = self._kernel_sizes
        return config


def create_cnn_micro(in_channels: int = 3) -> TinyCNN:
    """
    Create micro CNN variant (~5k params).

    Architecture: 2 conv layers with 16 filters each
    """
    return TinyCNN(
        in_channels=in_channels,
        channels=[16, 16],
        kernel_sizes=[3, 3],
        policy_hidden=16,
        value_hidden=16,
    )


def create_cnn_tiny(in_channels: int = 3) -> TinyCNN:
    """
    Create tiny CNN variant (~15k params).

    Architecture: 3 conv layers with 32 filters each
    """
    return TinyCNN(
        in_channels=in_channels,
        channels=[32, 32, 32],
        kernel_sizes=[3, 3, 3],
        policy_hidden=32,
        value_hidden=32,
    )


def create_cnn_small(in_channels: int = 3) -> TinyCNN:
    """
    Create small CNN variant (~50k params).

    Architecture: 4 conv layers with 64 filters each
    """
    return TinyCNN(
        in_channels=in_channels,
        channels=[64, 64, 64, 64],
        kernel_sizes=[3, 3, 3, 3],
        policy_hidden=64,
        value_hidden=64,
    )
