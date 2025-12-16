"""
Residual Network models for Connect Four.

CNNs with skip connections for deeper networks without gradient degradation.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .base import ConnectFourModel, PolicyValueHead


class ResidualBlock(nn.Module):
    """
    Basic residual block with two convolutional layers.

    Architecture: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        """
        Initialize the residual block.

        Args:
            channels: Number of input and output channels
            kernel_size: Kernel size for convolutions
        """
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.

        Args:
            x: Input tensor of shape (batch, channels, H, W)

        Returns:
            Output tensor with same shape as input
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        out = out + residual
        out = self.relu(out)

        return out


class TinyResNet(ConnectFourModel):
    """
    Configurable ResNet for Connect Four.

    Architecture:
    - Initial conv to expand channels
    - Stack of residual blocks
    - Global pooling -> PolicyValueHead

    Input shape: (batch, 3, 6, 7) - 3 channels for one-hot encoding

    Variants:
    - micro: 2 res blocks, 32 channels (~20k params)
    - tiny: 4 res blocks, 32 channels (~35k params)
    - small: 6 res blocks, 64 channels (~100k params)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 3,
        policy_hidden: int = 32,
        value_hidden: int = 32,
    ):
        """
        Initialize the ResNet.

        Args:
            in_channels: Number of input channels (3 for one-hot encoding)
            hidden_channels: Number of channels in residual blocks
            num_blocks: Number of residual blocks
            kernel_size: Kernel size for convolutions
            policy_hidden: Hidden dimension for policy head
            value_hidden: Hidden dimension for value head
        """
        super().__init__(input_size=126, input_shape=(in_channels, 6, 7))

        self._hidden_channels = hidden_channels
        self._num_blocks = num_blocks
        self._architecture_name = f"resnet-{num_blocks}b-{hidden_channels}c"

        # Initial convolution to expand channels
        padding = kernel_size // 2
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # Stack of residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_channels, kernel_size))
        self.res_blocks = nn.Sequential(*blocks)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Policy and value heads
        self.heads = PolicyValueHead(
            input_dim=hidden_channels,
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

        # Initial conv
        features = self.initial_conv(x)

        # Residual blocks
        features = self.res_blocks(features)

        # Global pooling
        features = self.global_pool(features)  # (batch, channels, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, channels)

        return self.heads(features)

    def get_config(self) -> dict:
        """Returns model configuration for serialization."""
        config = super().get_config()
        config["hidden_channels"] = self._hidden_channels
        config["num_blocks"] = self._num_blocks
        return config


def create_resnet_micro(in_channels: int = 3) -> TinyResNet:
    """
    Create micro ResNet variant (~20k params).

    Architecture: 2 residual blocks, 32 channels
    """
    return TinyResNet(
        in_channels=in_channels,
        hidden_channels=32,
        num_blocks=2,
        kernel_size=3,
        policy_hidden=32,
        value_hidden=32,
    )


def create_resnet_tiny(in_channels: int = 3) -> TinyResNet:
    """
    Create tiny ResNet variant (~35k params).

    Architecture: 4 residual blocks, 32 channels
    """
    return TinyResNet(
        in_channels=in_channels,
        hidden_channels=32,
        num_blocks=4,
        kernel_size=3,
        policy_hidden=32,
        value_hidden=32,
    )


def create_resnet_small(in_channels: int = 3) -> TinyResNet:
    """
    Create small ResNet variant (~100k params).

    Architecture: 6 residual blocks, 64 channels
    """
    return TinyResNet(
        in_channels=in_channels,
        hidden_channels=64,
        num_blocks=6,
        kernel_size=3,
        policy_hidden=64,
        value_hidden=64,
    )
