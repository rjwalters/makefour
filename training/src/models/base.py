"""
Base class for Connect Four neural network models.

All models inherit from this class and implement a common interface
for training, evaluation, and export.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class ConnectFourModel(nn.Module, ABC):
    """
    Abstract base class for Connect Four neural networks.

    All models must:
    - Accept encoded board positions as input
    - Output policy logits (7 values) and value estimate (1 value)
    - Support ONNX export
    """

    def __init__(self, input_size: int, input_shape: Tuple[int, ...] | None = None):
        """
        Initialize the model.

        Args:
            input_size: Total number of input features (e.g., 85 for flat-binary, 126 for one-hot)
            input_shape: Optional shape for reshaping input (e.g., (3, 6, 7) for CNN)
        """
        super().__init__()
        self.input_size = input_size
        self.input_shape = input_shape
        self._architecture_name = "base"

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, input_size) or (batch, *input_shape)

        Returns:
            Tuple of:
                - policy: Logits for each column (batch, 7)
                - value: Position evaluation (batch, 1) in range [-1, 1]
        """
        pass

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with softmax applied to policy.

        Args:
            x: Input tensor

        Returns:
            Tuple of:
                - policy: Probabilities for each column (batch, 7)
                - value: Position evaluation (batch, 1) in range [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self(x)
            policy = torch.softmax(policy_logits, dim=-1)
        return policy, value

    def param_count(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_string(self) -> str:
        """Returns human-readable architecture description."""
        return f"{self._architecture_name} ({self.param_count():,} params)"

    def get_config(self) -> dict:
        """Returns model configuration for serialization."""
        return {
            "architecture": self._architecture_name,
            "input_size": self.input_size,
            "input_shape": self.input_shape,
            "param_count": self.param_count(),
        }

    def summary(self) -> str:
        """Returns a summary of the model architecture."""
        lines = [
            f"Model: {self._architecture_name}",
            f"Input size: {self.input_size}",
            f"Input shape: {self.input_shape}",
            f"Parameters: {self.param_count():,}",
            "",
            "Layers:",
        ]

        for name, module in self.named_modules():
            if name and not "." in name:  # Only top-level modules
                param_count = sum(p.numel() for p in module.parameters())
                lines.append(f"  {name}: {module.__class__.__name__} ({param_count:,} params)")

        return "\n".join(lines)


class PolicyValueHead(nn.Module):
    """
    Shared policy and value head architecture.

    Takes features from a backbone and produces policy logits and value estimate.
    """

    def __init__(
        self,
        input_dim: int,
        policy_hidden: int = 32,
        value_hidden: int = 32,
    ):
        """
        Initialize policy and value heads.

        Args:
            input_dim: Dimension of input features from backbone
            policy_hidden: Hidden dimension for policy head
            value_hidden: Hidden dimension for value head
        """
        super().__init__()

        # Policy head: features -> hidden -> 7 logits
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, 7),
        )

        # Value head: features -> hidden -> 1 value with tanh
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy and value heads.

        Args:
            x: Feature tensor of shape (batch, input_dim)

        Returns:
            policy: Logits of shape (batch, 7)
            value: Value estimate of shape (batch, 1)
        """
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


def count_parameters(model: nn.Module) -> dict:
    """
    Count parameters by layer type.

    Returns:
        Dictionary with parameter counts by layer type
    """
    counts = {"total": 0, "trainable": 0}
    layer_types = {}

    for name, param in model.named_parameters():
        count = param.numel()
        counts["total"] += count
        if param.requires_grad:
            counts["trainable"] += count

        # Extract layer type from name
        layer_name = name.split(".")[0]
        if layer_name not in layer_types:
            layer_types[layer_name] = 0
        layer_types[layer_name] += count

    counts["by_layer"] = layer_types
    return counts
