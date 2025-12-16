"""
Model registry for Connect Four neural networks.

Provides a unified interface for creating models by name and listing
available architectures.
"""

from typing import Callable

from .base import ConnectFourModel
from .mlp import TinyMLP, create_mlp_micro, create_mlp_tiny, create_mlp_small
from .cnn import TinyCNN, create_cnn_micro, create_cnn_tiny, create_cnn_small
from .transformer import MicroTransformer, create_transformer_micro, create_transformer_tiny, create_transformer_small
from .resnet import TinyResNet, create_resnet_micro, create_resnet_tiny, create_resnet_small


# Registry of model factory functions
MODEL_REGISTRY: dict[str, Callable[[], ConnectFourModel]] = {
    # MLP variants
    "mlp-micro": create_mlp_micro,
    "mlp-tiny": create_mlp_tiny,
    "mlp-small": create_mlp_small,
    # CNN variants
    "cnn-micro": create_cnn_micro,
    "cnn-tiny": create_cnn_tiny,
    "cnn-small": create_cnn_small,
    # Transformer variants
    "transformer-micro": create_transformer_micro,
    "transformer-tiny": create_transformer_tiny,
    "transformer-small": create_transformer_small,
    # ResNet variants
    "resnet-micro": create_resnet_micro,
    "resnet-tiny": create_resnet_tiny,
    "resnet-small": create_resnet_small,
}

# Model info for documentation
MODEL_INFO: dict[str, dict] = {
    "mlp-micro": {
        "description": "Minimal MLP with single hidden layer",
        "architecture": "85 -> 32 -> heads",
        "params": "~3k",
        "input_encoding": "flat-binary",
    },
    "mlp-tiny": {
        "description": "Two-layer MLP",
        "architecture": "85 -> 64 -> 32 -> heads",
        "params": "~7k",
        "input_encoding": "flat-binary",
    },
    "mlp-small": {
        "description": "Three-layer MLP with wider layers",
        "architecture": "85 -> 128 -> 64 -> heads",
        "params": "~15k",
        "input_encoding": "flat-binary",
    },
    "cnn-micro": {
        "description": "Minimal CNN with 2 conv layers",
        "architecture": "2 conv layers, 16 filters",
        "params": "~5k",
        "input_encoding": "onehot-6x7x3",
    },
    "cnn-tiny": {
        "description": "CNN with 3 conv layers",
        "architecture": "3 conv layers, 32 filters",
        "params": "~15k",
        "input_encoding": "onehot-6x7x3",
    },
    "cnn-small": {
        "description": "CNN with 4 conv layers",
        "architecture": "4 conv layers, 64 filters",
        "params": "~50k",
        "input_encoding": "onehot-6x7x3",
    },
    "transformer-micro": {
        "description": "Minimal transformer encoder",
        "architecture": "2 layers, 32 dim, 2 heads",
        "params": "~10k",
        "input_encoding": "flat-binary",
    },
    "transformer-tiny": {
        "description": "Small transformer encoder",
        "architecture": "3 layers, 48 dim, 4 heads",
        "params": "~30k",
        "input_encoding": "flat-binary",
    },
    "transformer-small": {
        "description": "Medium transformer encoder",
        "architecture": "4 layers, 64 dim, 4 heads",
        "params": "~80k",
        "input_encoding": "flat-binary",
    },
    "resnet-micro": {
        "description": "Minimal ResNet with 2 blocks",
        "architecture": "2 res blocks, 32 channels",
        "params": "~20k",
        "input_encoding": "onehot-6x7x3",
    },
    "resnet-tiny": {
        "description": "Small ResNet with 4 blocks",
        "architecture": "4 res blocks, 32 channels",
        "params": "~35k",
        "input_encoding": "onehot-6x7x3",
    },
    "resnet-small": {
        "description": "Medium ResNet with 6 blocks",
        "architecture": "6 res blocks, 64 channels",
        "params": "~100k",
        "input_encoding": "onehot-6x7x3",
    },
}


def create_model(name: str, **kwargs) -> ConnectFourModel:
    """
    Create a model by name.

    Args:
        name: Model name (e.g., "mlp-tiny", "cnn-small")
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name is not recognized
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {name}. Available models: {available}")

    # Get factory function
    factory = MODEL_REGISTRY[name]

    # Create model
    if kwargs:
        # For custom kwargs, we need to instantiate the class directly
        # Determine the class from the factory
        if "mlp" in name:
            return TinyMLP(**kwargs)
        elif "cnn" in name:
            return TinyCNN(**kwargs)
        elif "transformer" in name:
            return MicroTransformer(**kwargs)
        elif "resnet" in name:
            return TinyResNet(**kwargs)
    else:
        return factory()

    return factory()


def list_models() -> list[str]:
    """
    List all available model names.

    Returns:
        Sorted list of model names
    """
    return sorted(MODEL_REGISTRY.keys())


def get_model_info(name: str) -> dict:
    """
    Get information about a model.

    Args:
        name: Model name

    Returns:
        Dictionary with model information

    Raises:
        ValueError: If model name is not recognized
    """
    if name not in MODEL_INFO:
        available = ", ".join(sorted(MODEL_INFO.keys()))
        raise ValueError(f"Unknown model: {name}. Available models: {available}")

    return MODEL_INFO[name].copy()


def print_model_table() -> str:
    """
    Generate a formatted table of all models.

    Returns:
        Formatted string table
    """
    lines = [
        "| Model | Description | Params | Encoding |",
        "|-------|-------------|--------|----------|",
    ]

    for name in sorted(MODEL_INFO.keys()):
        info = MODEL_INFO[name]
        lines.append(
            f"| {name} | {info['description']} | {info['params']} | {info['input_encoding']} |"
        )

    return "\n".join(lines)
