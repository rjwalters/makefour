"""
Metadata utilities for ONNX models.

Embeds and retrieves metadata in ONNX model files to track model
provenance, architecture, and training information.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelMetadata:
    """Metadata embedded in exported ONNX models."""

    # Model identification
    model_name: str
    architecture: str

    # Model characteristics
    param_count: int
    input_encoding: str
    input_shape: str  # e.g., "3,6,7" or "85"

    # Training information (optional)
    training_games: int | None = None
    training_epochs: int | None = None
    estimated_elo: int | None = None

    # Version and provenance
    makefour_version: str = "1.0.0"
    export_date: str | None = None
    checkpoint_path: str | None = None

    # Additional notes
    notes: str | None = None

    def __post_init__(self):
        if self.export_date is None:
            self.export_date = datetime.now().isoformat()

    def to_dict(self) -> dict[str, str]:
        """Convert to string dictionary for ONNX metadata."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = str(value)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ModelMetadata":
        """Create from string dictionary (ONNX metadata format)."""
        # Convert string values back to appropriate types
        converted = {}
        int_fields = {"param_count", "training_games", "training_epochs", "estimated_elo"}

        for key, value in data.items():
            if key in int_fields and value is not None and value != "None":
                converted[key] = int(value)
            elif value == "None":
                converted[key] = None
            else:
                converted[key] = value

        return cls(**converted)


def add_metadata(
    onnx_path: str | Path,
    metadata: ModelMetadata | dict[str, Any],
) -> None:
    """
    Add metadata to an ONNX model file.

    Args:
        onnx_path: Path to ONNX model file
        metadata: ModelMetadata instance or dictionary of metadata
    """
    import onnx

    # Load model
    model = onnx.load(str(onnx_path))

    # Convert to dictionary if needed
    if isinstance(metadata, ModelMetadata):
        meta_dict = metadata.to_dict()
    else:
        meta_dict = {k: str(v) for k, v in metadata.items() if v is not None}

    # Add metadata properties
    for key, value in meta_dict.items():
        meta_prop = model.metadata_props.add()
        meta_prop.key = key
        meta_prop.value = str(value)

    # Save model
    onnx.save(model, str(onnx_path))


def get_metadata(onnx_path: str | Path) -> dict[str, str]:
    """
    Retrieve metadata from an ONNX model file.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Dictionary of metadata key-value pairs
    """
    import onnx

    model = onnx.load(str(onnx_path))

    metadata = {}
    for prop in model.metadata_props:
        metadata[prop.key] = prop.value

    return metadata


def get_typed_metadata(onnx_path: str | Path) -> ModelMetadata | None:
    """
    Retrieve metadata as a typed ModelMetadata object.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        ModelMetadata if all required fields present, None otherwise
    """
    raw_metadata = get_metadata(onnx_path)

    # Check for required fields
    required_fields = {"model_name", "architecture", "param_count", "input_encoding", "input_shape"}
    if not required_fields.issubset(raw_metadata.keys()):
        return None

    try:
        return ModelMetadata.from_dict(raw_metadata)
    except (TypeError, ValueError):
        return None


def create_metadata_from_model(
    model,
    model_name: str,
    checkpoint_path: str | Path | None = None,
    **kwargs,
) -> ModelMetadata:
    """
    Create metadata from a PyTorch model.

    Args:
        model: ConnectFourModel instance
        model_name: Name for the model (e.g., "cnn-tiny-v1")
        checkpoint_path: Optional path to the checkpoint used
        **kwargs: Additional metadata fields

    Returns:
        ModelMetadata instance
    """
    from ..models import MODEL_INFO

    # Get architecture info
    architecture = model._architecture_name if hasattr(model, "_architecture_name") else "unknown"

    # Get input encoding from registry or model
    input_encoding = "unknown"
    for name, info in MODEL_INFO.items():
        if name in model_name or architecture in name:
            input_encoding = info.get("input_encoding", "unknown")
            break

    # Determine input shape string
    if model.input_shape is not None:
        input_shape = ",".join(str(d) for d in model.input_shape)
    else:
        input_shape = str(model.input_size)

    return ModelMetadata(
        model_name=model_name,
        architecture=architecture,
        param_count=model.param_count(),
        input_encoding=input_encoding,
        input_shape=input_shape,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        **kwargs,
    )


def print_metadata(onnx_path: str | Path) -> str:
    """
    Generate a formatted string of model metadata.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Formatted metadata string
    """
    metadata = get_metadata(onnx_path)

    if not metadata:
        return "No metadata found in model"

    lines = ["Model Metadata:", "-" * 40]

    # Group related fields
    groups = {
        "Identification": ["model_name", "architecture"],
        "Model Characteristics": ["param_count", "input_encoding", "input_shape"],
        "Training": ["training_games", "training_epochs", "estimated_elo"],
        "Provenance": ["makefour_version", "export_date", "checkpoint_path"],
        "Notes": ["notes"],
    }

    for group_name, fields in groups.items():
        group_items = [(f, metadata.get(f)) for f in fields if f in metadata]
        if group_items:
            lines.append(f"\n{group_name}:")
            for key, value in group_items:
                lines.append(f"  {key}: {value}")

    # Any remaining fields
    known_fields = {f for fields in groups.values() for f in fields}
    other_fields = [(k, v) for k, v in metadata.items() if k not in known_fields]
    if other_fields:
        lines.append("\nOther:")
        for key, value in other_fields:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)
