#!/usr/bin/env python3
"""
ONNX Export CLI for Connect Four Models

Export trained PyTorch models to ONNX format for deployment.

Usage:
    # Export a model by name (creates untrained model)
    python scripts/export.py --model cnn-tiny --output models/cnn-tiny.onnx

    # Export from checkpoint
    python scripts/export.py --checkpoint checkpoints/cnn-tiny-epoch50.pt --output models/cnn-tiny.onnx

    # Export with optimization
    python scripts/export.py --checkpoint checkpoints/model.pt --output models/model.onnx --optimize

    # Export with quantization for smaller size
    python scripts/export.py --checkpoint checkpoints/model.pt --output models/model.onnx --quantize

    # Batch export all checkpoints in a directory
    python scripts/export.py --checkpoint-dir checkpoints/ --output-dir models/ --all

    # Validate an exported model
    python scripts/export.py --validate models/cnn-tiny.onnx --model cnn-tiny

    # Show model info
    python scripts/export.py --info models/cnn-tiny.onnx
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Export Connect Four models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Export options
    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument(
        "--model",
        type=str,
        help="Model name from registry (e.g., cnn-tiny, mlp-small)",
    )
    export_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (.pt file)",
    )
    export_group.add_argument(
        "--output",
        type=str,
        help="Output path for ONNX file",
    )

    # Batch export
    batch_group = parser.add_argument_group("Batch Export")
    batch_group.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing checkpoints for batch export",
    )
    batch_group.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch export",
    )
    batch_group.add_argument(
        "--all",
        action="store_true",
        help="Export all checkpoints in --checkpoint-dir",
    )

    # Optimization options
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument(
        "--optimize",
        action="store_true",
        help="Apply ONNX optimizations",
    )
    opt_group.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization (reduces size ~4x)",
    )
    opt_group.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )

    # Validation
    val_group = parser.add_argument_group("Validation")
    val_group.add_argument(
        "--validate",
        type=str,
        metavar="ONNX_PATH",
        help="Validate an exported ONNX model",
    )
    val_group.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-export validation",
    )

    # Info
    info_group = parser.add_argument_group("Information")
    info_group.add_argument(
        "--info",
        type=str,
        metavar="ONNX_PATH",
        help="Show metadata from an ONNX model",
    )
    info_group.add_argument(
        "--list-models",
        action="store_true",
        help="List available model architectures",
    )

    # Metadata
    meta_group = parser.add_argument_group("Metadata")
    meta_group.add_argument(
        "--training-games",
        type=int,
        help="Number of training games (for metadata)",
    )
    meta_group.add_argument(
        "--training-epochs",
        type=int,
        help="Number of training epochs (for metadata)",
    )
    meta_group.add_argument(
        "--estimated-elo",
        type=int,
        help="Estimated Elo rating (for metadata)",
    )

    args = parser.parse_args()

    # Handle info commands
    if args.list_models:
        list_models()
        return

    if args.info:
        show_model_info(args.info)
        return

    # Handle validation-only mode
    if args.validate:
        if not args.model:
            parser.error("--validate requires --model to specify architecture")
        validate_model(args.validate, args.model)
        return

    # Handle batch export
    if args.all and args.checkpoint_dir:
        batch_export(args)
        return

    # Handle single export
    if args.model or args.checkpoint:
        single_export(args)
        return

    parser.print_help()


def list_models():
    """List all available model architectures."""
    from models import list_models as get_models, get_model_info

    print("Available Models:")
    print("-" * 60)

    for name in get_models():
        info = get_model_info(name)
        print(f"\n{name}")
        print(f"  Description: {info['description']}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Parameters: {info['params']}")
        print(f"  Input Encoding: {info['input_encoding']}")


def show_model_info(onnx_path: str):
    """Display metadata from an ONNX model."""
    from export.metadata import print_metadata

    path = Path(onnx_path)
    if not path.exists():
        print(f"Error: File not found: {onnx_path}")
        sys.exit(1)

    print(f"File: {onnx_path}")
    print(f"Size: {path.stat().st_size / 1024:.1f} KB")
    print()
    print(print_metadata(onnx_path))


def validate_model(onnx_path: str, model_name: str):
    """Validate an ONNX model against its PyTorch equivalent."""
    from models import create_model
    from export import validate_onnx_model
    from export.onnx_export import get_input_shape_for_model

    print(f"Validating {onnx_path}...")

    # Create model
    model = create_model(model_name)
    input_shape = get_input_shape_for_model(model)

    # Validate
    result = validate_onnx_model(onnx_path, model, input_shape)

    print(result)

    if not result.valid:
        sys.exit(1)


def single_export(args):
    """Export a single model."""
    import torch
    from models import create_model
    from export import export_to_onnx, validate_onnx_model, ExportConfig
    from export.metadata import add_metadata, create_metadata_from_model
    from export.onnx_export import get_input_shape_for_model

    # Determine model name
    if args.model:
        model_name = args.model
    elif args.checkpoint:
        # Try to infer from checkpoint filename
        checkpoint_name = Path(args.checkpoint).stem
        # Common patterns: model-name-epochN, model-name_epochN
        for sep in ["-epoch", "_epoch", "-e", "_e"]:
            if sep in checkpoint_name:
                model_name = checkpoint_name.split(sep)[0]
                break
        else:
            print("Error: Cannot infer model name from checkpoint. Use --model")
            sys.exit(1)
    else:
        print("Error: Must specify --model or --checkpoint")
        sys.exit(1)

    # Create model
    print(f"Creating model: {model_name}")
    try:
        model = create_model(model_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("models") / f"{model_name}.onnx"

    print(f"Output: {output_path}")

    # Create export config
    config = ExportConfig(
        output_path=output_path,
        opset_version=args.opset,
        optimize=args.optimize,
        quantize=args.quantize,
    )

    # Export
    print("Exporting to ONNX...")
    result = export_to_onnx(model, config)

    print(f"  Size: {result.model_size_kb:.1f} KB")
    print(f"  Input shape: {result.input_shape}")
    print(f"  Optimized: {result.optimized}")
    print(f"  Quantized: {result.quantized}")

    # Add metadata
    print("Adding metadata...")
    metadata = create_metadata_from_model(
        model,
        model_name,
        checkpoint_path=args.checkpoint,
        training_games=args.training_games,
        training_epochs=args.training_epochs,
        estimated_elo=args.estimated_elo,
    )
    add_metadata(output_path, metadata)

    # Validate unless disabled
    if not args.no_validate:
        print("Validating...")
        input_shape = get_input_shape_for_model(model)
        val_result = validate_onnx_model(output_path, model, input_shape, num_samples=50)
        print(f"  {val_result}")

        if not val_result.valid:
            print("Warning: Validation failed!")
            sys.exit(1)

    print(f"\nExport complete: {output_path}")


def batch_export(args):
    """Export all checkpoints in a directory."""
    import torch
    from models import create_model, list_models as get_models
    from export import export_to_onnx, ExportConfig
    from export.metadata import add_metadata, create_metadata_from_model

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("models")

    if not checkpoint_dir.exists():
        print(f"Error: Directory not found: {checkpoint_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    available_models = set(get_models())

    exported = 0
    failed = 0

    for checkpoint_path in sorted(checkpoints):
        checkpoint_name = checkpoint_path.stem

        # Try to infer model name
        model_name = None
        for model in available_models:
            if checkpoint_name.startswith(model):
                model_name = model
                break

        if model_name is None:
            print(f"  Skipping {checkpoint_name}: Cannot determine model type")
            continue

        print(f"\nExporting {checkpoint_name} ({model_name})...")

        try:
            # Create and load model
            model = create_model(model_name)
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            # Export
            output_path = output_dir / f"{checkpoint_name}.onnx"
            config = ExportConfig(
                output_path=output_path,
                optimize=args.optimize,
                quantize=args.quantize,
            )
            result = export_to_onnx(model, config)

            # Add metadata
            metadata = create_metadata_from_model(
                model,
                checkpoint_name,
                checkpoint_path=str(checkpoint_path),
            )
            add_metadata(output_path, metadata)

            print(f"  -> {output_path} ({result.model_size_kb:.1f} KB)")
            exported += 1

        except Exception as e:
            print(f"  Error: {e}")
            failed += 1

    print(f"\nBatch export complete: {exported} exported, {failed} failed")


if __name__ == "__main__":
    main()
