"""
Registry Update Script

Updates the model registry files in the MakeFour application
with metadata for newly trained models.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class RegistryConfig:
    """Configuration for registry update."""

    # Model metadata
    model_id: str
    name: str
    version: str
    architecture: str  # 'mlp', 'cnn', 'transformer'
    expected_elo: int
    size_bytes: int
    url: str
    encoding: str  # 'onehot-6x7x3', 'bitboard', 'flat-binary'

    # Training metadata
    training_games: int = 0
    training_epochs: int = 0
    training_date: str = ""

    # Paths to registry files
    client_registry_path: Path = field(
        default_factory=lambda: Path("src/ai/neural/loader.ts")
    )
    server_registry_path: Path = field(
        default_factory=lambda: Path("functions/api/models/index.ts")
    )


@dataclass
class ModelMetadata:
    """Model metadata matching the TypeScript interface."""

    id: str
    name: str
    architecture: str
    expected_elo: int
    size_bytes: int
    url: str
    version: str
    encoding: str
    training: Optional[dict] = None


def generate_model_metadata(config: RegistryConfig) -> ModelMetadata:
    """Generate ModelMetadata from config."""
    training = None
    if config.training_games > 0:
        training = {
            "games": config.training_games,
            "epochs": config.training_epochs,
            "date": config.training_date or datetime.now().strftime("%Y-%m-%d"),
        }

    return ModelMetadata(
        id=config.model_id,
        name=config.name,
        architecture=config.architecture,
        expected_elo=config.expected_elo,
        size_bytes=config.size_bytes,
        url=config.url,
        version=config.version,
        encoding=config.encoding,
        training=training,
    )


def format_typescript_metadata(metadata: ModelMetadata) -> str:
    """Format model metadata as TypeScript object literal."""
    lines = [
        "  {",
        f"    id: '{metadata.id}',",
        f"    name: '{metadata.name}',",
        f"    architecture: '{metadata.architecture}',",
        f"    expectedElo: {metadata.expected_elo},",
        f"    sizeBytes: {metadata.size_bytes},",
        f"    url: '{metadata.url}',",
        f"    version: '{metadata.version}',",
        f"    encoding: '{metadata.encoding}',",
    ]

    if metadata.training:
        lines.append("    training: {")
        lines.append(f"      games: {metadata.training['games']},")
        lines.append(f"      epochs: {metadata.training['epochs']},")
        lines.append(f"      date: '{metadata.training['date']}',")
        lines.append("    },")

    lines.append("  },")
    return "\n".join(lines)


def update_typescript_registry(
    file_path: Path,
    metadata: ModelMetadata,
    registry_var_name: str = "MODEL_REGISTRY",
) -> tuple[bool, str]:
    """
    Update a TypeScript registry file with new model metadata.

    Args:
        file_path: Path to the TypeScript file
        metadata: Model metadata to add
        registry_var_name: Name of the registry array variable

    Returns:
        Tuple of (success, error_message)
    """
    if not file_path.exists():
        return False, f"Registry file not found: {file_path}"

    content = file_path.read_text()

    # Check if model already exists
    if f"id: '{metadata.id}'" in content:
        return False, f"Model '{metadata.id}' already exists in registry"

    # Find the registry array
    # Pattern: const MODEL_REGISTRY: ModelMetadata[] = [
    # or: const MODEL_REGISTRY = [
    pattern = rf"(const\s+{registry_var_name}[^=]*=\s*\[)"

    match = re.search(pattern, content)
    if not match:
        return False, f"Could not find {registry_var_name} array in {file_path}"

    # Find where to insert (after the first model entry or after the opening bracket)
    array_start = match.end()

    # Look for the comment about future models
    future_comment_pattern = r"(\s*//\s*Future models will be added here[^\n]*\n)"
    future_match = re.search(future_comment_pattern, content[array_start:])

    if future_match:
        # Insert before the comment
        insert_pos = array_start + future_match.start()
        new_entry = format_typescript_metadata(metadata) + "\n"
        new_content = content[:insert_pos] + new_entry + content[insert_pos:]
    else:
        # Find the closing bracket of the array
        bracket_depth = 1
        pos = array_start
        while bracket_depth > 0 and pos < len(content):
            if content[pos] == "[":
                bracket_depth += 1
            elif content[pos] == "]":
                bracket_depth -= 1
            pos += 1

        if bracket_depth != 0:
            return False, f"Could not find closing bracket of {registry_var_name}"

        # Insert before the closing bracket
        insert_pos = pos - 1
        # Find the last non-whitespace before the bracket
        while insert_pos > array_start and content[insert_pos - 1] in " \t\n":
            insert_pos -= 1

        new_entry = "\n" + format_typescript_metadata(metadata)
        new_content = content[:insert_pos] + new_entry + content[insert_pos:]

    file_path.write_text(new_content)
    return True, ""


def update_registry(config: RegistryConfig) -> dict[str, Any]:
    """
    Update model registry files with new model metadata.

    Args:
        config: Registry configuration

    Returns:
        Dictionary with results for each file
    """
    metadata = generate_model_metadata(config)
    results: dict[str, Any] = {"metadata": metadata.__dict__, "files": {}}

    # Update client-side registry
    if config.client_registry_path.exists():
        success, error = update_typescript_registry(
            config.client_registry_path, metadata
        )
        results["files"]["client"] = {"success": success, "error": error}
    else:
        results["files"]["client"] = {
            "success": False,
            "error": f"File not found: {config.client_registry_path}",
        }

    # Update server-side registry
    if config.server_registry_path.exists():
        success, error = update_typescript_registry(
            config.server_registry_path, metadata
        )
        results["files"]["server"] = {"success": success, "error": error}
    else:
        results["files"]["server"] = {
            "success": False,
            "error": f"File not found: {config.server_registry_path}",
        }

    return results


def load_experiment_metadata(experiment_path: Path) -> dict[str, Any]:
    """
    Load training metadata from an experiment directory.

    Expects experiment_path to contain:
    - config.json: Training configuration
    - eval.json: Evaluation results
    """
    metadata: dict[str, Any] = {}

    config_file = experiment_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            metadata["architecture"] = config.get("model", {}).get("architecture", "mlp")
            metadata["encoding"] = config.get("model", {}).get("encoding", "flat-binary")
            metadata["training_epochs"] = config.get("training", {}).get("epochs", 0)

    eval_file = experiment_path / "eval.json"
    if eval_file.exists():
        with open(eval_file) as f:
            eval_data = json.load(f)
            metadata["expected_elo"] = eval_data.get("estimated_elo", 1200)
            metadata["training_games"] = eval_data.get("games_played", 0)

    return metadata


def main():
    """CLI entry point for registry update."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update model registry with new model metadata"
    )
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--name", required=True, help="Human-readable model name")
    parser.add_argument("--version", default="1.0.0", help="Model version")
    parser.add_argument(
        "--architecture",
        choices=["mlp", "cnn", "transformer"],
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--encoding",
        choices=["onehot-6x7x3", "bitboard", "flat-binary"],
        default="flat-binary",
        help="Position encoding type",
    )
    parser.add_argument("--expected-elo", type=int, required=True, help="Expected ELO")
    parser.add_argument("--size-bytes", type=int, required=True, help="Model file size")
    parser.add_argument("--url", required=True, help="Model download URL")

    # Training metadata
    parser.add_argument(
        "--training-games", type=int, default=0, help="Number of training games"
    )
    parser.add_argument(
        "--training-epochs", type=int, default=0, help="Number of training epochs"
    )
    parser.add_argument("--training-date", default="", help="Training date (YYYY-MM-DD)")

    # Experiment metadata (alternative to individual args)
    parser.add_argument("--experiment", help="Path to experiment directory")

    # Registry paths
    parser.add_argument(
        "--client-registry",
        default="src/ai/neural/loader.ts",
        help="Path to client-side registry",
    )
    parser.add_argument(
        "--server-registry",
        default="functions/api/models/index.ts",
        help="Path to server-side registry",
    )

    # Upload result (from upload.py)
    parser.add_argument(
        "--upload-result", help="Path to upload result JSON file"
    )

    args = parser.parse_args()

    # Load upload result if provided
    url = args.url
    size_bytes = args.size_bytes
    if args.upload_result:
        upload_path = Path(args.upload_result)
        if upload_path.exists():
            with open(upload_path) as f:
                upload_data = json.load(f)
                url = upload_data.get("url", url)
                size_bytes = upload_data.get("size_bytes", size_bytes)

    # Load experiment metadata if provided
    training_games = args.training_games
    training_epochs = args.training_epochs
    architecture = args.architecture
    encoding = args.encoding
    expected_elo = args.expected_elo

    if args.experiment:
        exp_metadata = load_experiment_metadata(Path(args.experiment))
        architecture = exp_metadata.get("architecture", architecture)
        encoding = exp_metadata.get("encoding", encoding)
        training_games = exp_metadata.get("training_games", training_games)
        training_epochs = exp_metadata.get("training_epochs", training_epochs)
        expected_elo = exp_metadata.get("expected_elo", expected_elo)

    config = RegistryConfig(
        model_id=args.model_id,
        name=args.name,
        version=args.version,
        architecture=architecture,
        expected_elo=expected_elo,
        size_bytes=size_bytes,
        url=url,
        encoding=encoding,
        training_games=training_games,
        training_epochs=training_epochs,
        training_date=args.training_date,
        client_registry_path=Path(args.client_registry),
        server_registry_path=Path(args.server_registry),
    )

    print(f"Updating registry with model: {config.model_id}")
    results = update_registry(config)

    print("\nResults:")
    print(f"  Model ID: {config.model_id}")
    print(f"  Name: {config.name}")
    print(f"  Architecture: {config.architecture}")
    print(f"  Expected ELO: {config.expected_elo}")
    print(f"  URL: {config.url}")

    all_success = True
    for file_type, result in results["files"].items():
        if result["success"]:
            print(f"  {file_type.capitalize()} registry: Updated successfully")
        else:
            print(f"  {file_type.capitalize()} registry: Failed - {result['error']}")
            all_success = False

    if not all_success:
        exit(1)


if __name__ == "__main__":
    main()
