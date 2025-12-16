"""
ONNX export functionality for Connect Four models.

Converts trained PyTorch models to ONNX format for deployment in:
- Cloudflare Workers (server-side via ONNX Runtime)
- Browser (client-side via ONNX Runtime Web)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from ..models import ConnectFourModel, MODEL_INFO


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""

    # Output settings
    output_path: str | Path
    opset_version: int = 14

    # Optimization settings
    optimize: bool = True
    quantize: bool = False

    # Input specification (auto-detected if not provided)
    input_shape: Tuple[int, ...] | None = None
    input_name: str = "board"
    output_names: list[str] = field(default_factory=lambda: ["policy", "value"])

    # Dynamic axes for batch size
    dynamic_batch: bool = True


@dataclass
class ExportResult:
    """Result of ONNX export operation."""

    output_path: Path
    model_size_bytes: int
    input_shape: Tuple[int, ...]
    opset_version: int
    optimized: bool
    quantized: bool

    @property
    def model_size_kb(self) -> float:
        """Model size in kilobytes."""
        return self.model_size_bytes / 1024

    @property
    def model_size_mb(self) -> float:
        """Model size in megabytes."""
        return self.model_size_bytes / (1024 * 1024)


def get_input_shape_for_model(model: ConnectFourModel) -> Tuple[int, ...]:
    """
    Determine the correct input shape for a model.

    Args:
        model: ConnectFourModel instance

    Returns:
        Input shape tuple (without batch dimension)
    """
    if model.input_shape is not None:
        # CNN/ResNet models have explicit 3D shape
        return model.input_shape
    else:
        # MLP/Transformer models use flat input
        return (model.input_size,)


def export_to_onnx(
    model: nn.Module,
    config: ExportConfig,
) -> ExportResult:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model (must be a ConnectFourModel)
        config: Export configuration

    Returns:
        ExportResult with export details

    Raises:
        ValueError: If model cannot be exported
        RuntimeError: If export or validation fails
    """
    import onnx

    # Ensure model is in eval mode
    model.eval()

    # Determine input shape
    if config.input_shape is not None:
        input_shape = config.input_shape
    elif isinstance(model, ConnectFourModel):
        input_shape = get_input_shape_for_model(model)
    else:
        raise ValueError(
            "input_shape must be provided for non-ConnectFourModel models"
        )

    # Create dummy input with batch dimension
    dummy_input = torch.randn(1, *input_shape)

    # Ensure output path is a Path object
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dynamic axes configuration
    dynamic_axes = None
    if config.dynamic_batch:
        dynamic_axes = {
            config.input_name: {0: "batch"},
        }
        for output_name in config.output_names:
            dynamic_axes[output_name] = {0: "batch"}

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=config.opset_version,
            do_constant_folding=True,
            input_names=[config.input_name],
            output_names=config.output_names,
            dynamic_axes=dynamic_axes,
        )

    # Validate the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Apply optimizations if requested
    optimized = False
    if config.optimize:
        optimized = _optimize_onnx_model(output_path)

    # Apply quantization if requested
    quantized = False
    if config.quantize:
        quantized = _quantize_onnx_model(output_path)

    # Get file size
    model_size = output_path.stat().st_size

    return ExportResult(
        output_path=output_path,
        model_size_bytes=model_size,
        input_shape=input_shape,
        opset_version=config.opset_version,
        optimized=optimized,
        quantized=quantized,
    )


def _optimize_onnx_model(model_path: Path) -> bool:
    """
    Apply ONNX optimizations to reduce model size and improve inference speed.

    Args:
        model_path: Path to ONNX model file

    Returns:
        True if optimization was applied successfully
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer

        # Load and optimize
        optimized_model = optimizer.optimize_model(str(model_path))

        # Save back to same path
        optimized_model.save_model_to_file(str(model_path))

        return True
    except ImportError:
        # onnxruntime.transformers not available
        return False
    except Exception:
        # Optimization failed but original model is still valid
        return False


def _quantize_onnx_model(model_path: Path) -> bool:
    """
    Apply dynamic INT8 quantization to reduce model size.

    Args:
        model_path: Path to ONNX model file

    Returns:
        True if quantization was applied successfully
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        # Create quantized output path
        quantized_path = model_path.with_suffix(".int8.onnx")

        # Apply dynamic quantization
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QUInt8,
        )

        # Replace original with quantized version
        quantized_path.replace(model_path)

        return True
    except ImportError:
        # onnxruntime.quantization not available
        return False
    except Exception:
        # Quantization failed but original model is still valid
        return False


def export_model_by_name(
    model_name: str,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    **export_kwargs,
) -> ExportResult:
    """
    Export a model by its registry name.

    Args:
        model_name: Model name from registry (e.g., "cnn-tiny")
        checkpoint_path: Optional path to trained weights
        output_path: Output path for ONNX file (defaults to models/{name}.onnx)
        **export_kwargs: Additional arguments passed to ExportConfig

    Returns:
        ExportResult with export details
    """
    from ..models import create_model

    # Create model
    model = create_model(model_name)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Determine output path
    if output_path is None:
        output_path = Path("models") / f"{model_name}.onnx"

    # Create export config
    config = ExportConfig(output_path=output_path, **export_kwargs)

    return export_to_onnx(model, config)


def get_encoding_for_model(model_name: str) -> str:
    """
    Get the input encoding type for a model.

    Args:
        model_name: Model name from registry

    Returns:
        Encoding type string ("onehot-6x7x3" or "flat-binary")
    """
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]["input_encoding"]

    # Default based on model prefix
    if model_name.startswith(("cnn", "resnet")):
        return "onehot-6x7x3"
    else:
        return "flat-binary"
