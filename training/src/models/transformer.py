"""
Transformer-based models for Connect Four.

Treats board positions as sequences for attention-based processing.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from .base import ConnectFourModel, PolicyValueHead


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence positions.
    """

    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :]


class MicroTransformer(ConnectFourModel):
    """
    Minimal transformer for Connect Four.

    Treats the 42 board positions as a sequence, with each position
    having an embedding that encodes the piece state (empty, P1, P2).

    Architecture:
    - Position embedding: 42 positions -> d_model dimensions
    - Piece embedding: 3 states (empty, P1, P2) -> d_model dimensions
    - Transformer encoder layers
    - Global pooling -> PolicyValueHead
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        policy_hidden: int = 32,
        value_hidden: int = 32,
    ):
        """
        Initialize the transformer.

        Args:
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            policy_hidden: Hidden dimension for policy head
            value_hidden: Hidden dimension for value head
        """
        # Input is flat board representation
        super().__init__(input_size=85, input_shape=(6, 7))

        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._architecture_name = f"transformer-{n_layers}L-{d_model}d-{n_heads}h"

        # Embeddings
        # Piece embedding: 3 states (empty=0, P1=1, P2=2)
        self.piece_embedding = nn.Embedding(3, d_model)

        # Positional encoding for 42 positions
        self.pos_encoding = PositionalEncoding(d_model, max_len=42)

        # Current player embedding (added to all positions)
        self.player_embedding = nn.Embedding(2, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)

        # Heads take pooled transformer output
        self.heads = PolicyValueHead(
            input_dim=d_model,
            policy_hidden=policy_hidden,
            value_hidden=value_hidden,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch, 85) for flat-binary encoding
               First 42: P1 pieces, Next 42: P2 pieces, Last 1: current player

        Returns:
            Tuple of:
                - policy: Logits for each column (batch, 7)
                - value: Position evaluation (batch, 1) in range [-1, 1]
        """
        batch_size = x.size(0)

        # Parse flat-binary encoding
        # First 42 bits: P1 pieces, next 42 bits: P2 pieces
        p1_pieces = x[:, :42]  # (batch, 42)
        p2_pieces = x[:, 42:84]  # (batch, 42)
        current_player = x[:, 84].long()  # (batch,)

        # Create piece indices: 0=empty, 1=P1, 2=P2
        piece_indices = torch.zeros(batch_size, 42, dtype=torch.long, device=x.device)
        piece_indices = piece_indices + p1_pieces.long() + 2 * p2_pieces.long()

        # Get piece embeddings
        embeddings = self.piece_embedding(piece_indices)  # (batch, 42, d_model)

        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)

        # Add current player embedding to all positions
        player_emb = self.player_embedding(current_player)  # (batch, d_model)
        embeddings = embeddings + player_emb.unsqueeze(1)

        # Apply transformer
        features = self.transformer(embeddings)  # (batch, 42, d_model)

        # Global average pooling
        features = features.mean(dim=1)  # (batch, d_model)

        # Layer norm
        features = self.output_norm(features)

        return self.heads(features)

    def get_config(self) -> dict:
        """Returns model configuration for serialization."""
        config = super().get_config()
        config["d_model"] = self._d_model
        config["n_heads"] = self._n_heads
        config["n_layers"] = self._n_layers
        return config


def create_transformer_micro() -> MicroTransformer:
    """
    Create micro transformer variant (~10k params).

    Architecture: 2 layers, 32 dim, 2 heads
    """
    return MicroTransformer(
        d_model=32,
        n_heads=2,
        n_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        policy_hidden=32,
        value_hidden=32,
    )


def create_transformer_tiny() -> MicroTransformer:
    """
    Create tiny transformer variant (~30k params).

    Architecture: 3 layers, 48 dim, 4 heads
    """
    return MicroTransformer(
        d_model=48,
        n_heads=4,
        n_layers=3,
        dim_feedforward=96,
        dropout=0.1,
        policy_hidden=48,
        value_hidden=48,
    )


def create_transformer_small() -> MicroTransformer:
    """
    Create small transformer variant (~80k params).

    Architecture: 4 layers, 64 dim, 4 heads
    """
    return MicroTransformer(
        d_model=64,
        n_heads=4,
        n_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        policy_hidden=64,
        value_hidden=64,
    )
