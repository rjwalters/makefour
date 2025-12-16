"""ONNX export pipeline for Connect Four neural network models."""

from .onnx_export import (
    export_to_onnx,
    ExportConfig,
    ExportResult,
)
from .validation import (
    validate_onnx_model,
    compare_outputs,
    ValidationResult,
)
from .metadata import (
    add_metadata,
    get_metadata,
    ModelMetadata,
)

__all__ = [
    # Export
    "export_to_onnx",
    "ExportConfig",
    "ExportResult",
    # Validation
    "validate_onnx_model",
    "compare_outputs",
    "ValidationResult",
    # Metadata
    "add_metadata",
    "get_metadata",
    "ModelMetadata",
]
