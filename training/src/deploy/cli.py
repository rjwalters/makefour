"""
Model Deployment CLI

Unified CLI for model deployment workflow:
1. Export model to ONNX
2. Upload to cloud storage
3. Update registry
4. (Optional) Create PR
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .upload import ModelUploadConfig, StorageBackend, upload_model
from .registry_update import RegistryConfig, update_registry, load_experiment_metadata


def export_model(
    checkpoint_path: Path,
    output_path: Path,
    model_config: Optional[Path] = None,
) -> bool:
    """
    Export a PyTorch checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path for output ONNX file
        model_config: Optional path to model configuration

    Returns:
        True if export succeeded
    """
    # Import here to avoid requiring torch when not exporting
    try:
        from ..data.export import export_model_to_onnx
    except ImportError:
        print("Error: Could not import export module. Are training dependencies installed?")
        return False

    try:
        export_model_to_onnx(checkpoint_path, output_path, model_config)
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def full_deploy(
    model_path: Path,
    model_id: str,
    name: str,
    version: str,
    architecture: str,
    encoding: str,
    expected_elo: int,
    bucket: str,
    backend: StorageBackend,
    makefour_root: Path,
    training_games: int = 0,
    training_epochs: int = 0,
    create_pr: bool = False,
    pr_branch: Optional[str] = None,
) -> dict:
    """
    Execute full deployment workflow.

    Returns:
        Dictionary with deployment results
    """
    results = {
        "upload": None,
        "registry": None,
        "pr": None,
    }

    # Step 1: Upload model
    print(f"\n1. Uploading model to {backend.value}://{bucket}...")
    upload_config = ModelUploadConfig(
        model_path=model_path,
        model_id=model_id,
        version=version,
        bucket=bucket,
        backend=backend,
    )
    upload_result = upload_model(upload_config)
    results["upload"] = {
        "success": upload_result.success,
        "url": upload_result.url,
        "size_bytes": upload_result.size_bytes,
        "error": upload_result.error,
    }

    if not upload_result.success:
        print(f"   Upload failed: {upload_result.error}")
        return results

    print(f"   Uploaded to: {upload_result.url}")

    # Step 2: Update registry
    print("\n2. Updating model registry...")
    registry_config = RegistryConfig(
        model_id=model_id,
        name=name,
        version=version,
        architecture=architecture,
        expected_elo=expected_elo,
        size_bytes=upload_result.size_bytes,
        url=upload_result.url,
        encoding=encoding,
        training_games=training_games,
        training_epochs=training_epochs,
        training_date=datetime.now().strftime("%Y-%m-%d"),
        client_registry_path=makefour_root / "src/ai/neural/loader.ts",
        server_registry_path=makefour_root / "functions/api/models/index.ts",
    )

    registry_results = update_registry(registry_config)
    results["registry"] = registry_results

    for file_type, result in registry_results["files"].items():
        if result["success"]:
            print(f"   {file_type.capitalize()} registry: Updated")
        else:
            print(f"   {file_type.capitalize()} registry: Failed - {result['error']}")

    # Step 3: Create PR (if requested)
    if create_pr:
        print("\n3. Creating pull request...")
        pr_result = create_deployment_pr(
            makefour_root=makefour_root,
            model_id=model_id,
            branch=pr_branch or f"deploy-{model_id}",
        )
        results["pr"] = pr_result

        if pr_result["success"]:
            print(f"   PR created: {pr_result['url']}")
        else:
            print(f"   PR creation failed: {pr_result['error']}")

    return results


def create_deployment_pr(
    makefour_root: Path,
    model_id: str,
    branch: str,
) -> dict:
    """
    Create a git branch and PR for the deployment.

    Returns:
        Dictionary with PR creation results
    """
    try:
        # Create branch
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=makefour_root,
            check=True,
            capture_output=True,
        )

        # Stage changes
        subprocess.run(
            ["git", "add", "src/ai/neural/loader.ts", "functions/api/models/index.ts"],
            cwd=makefour_root,
            check=True,
            capture_output=True,
        )

        # Commit
        commit_msg = f"feat: add {model_id} to model registry"
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=makefour_root,
            check=True,
            capture_output=True,
        )

        # Push
        subprocess.run(
            ["git", "push", "-u", "origin", branch],
            cwd=makefour_root,
            check=True,
            capture_output=True,
        )

        # Create PR using gh CLI
        pr_body = f"""## Summary

Add {model_id} to the neural network model registry.

## Changes

- Added model metadata to client-side registry (src/ai/neural/loader.ts)
- Added model metadata to server-side registry (functions/api/models/index.ts)

## Test Plan

- [ ] Verify model can be loaded from the registry
- [ ] Verify model inference works correctly
"""
        result = subprocess.run(
            [
                "gh", "pr", "create",
                "--title", f"Add {model_id} to model registry",
                "--body", pr_body,
                "--label", "loom:review-requested",
            ],
            cwd=makefour_root,
            check=True,
            capture_output=True,
            text=True,
        )

        # Extract PR URL from output
        pr_url = result.stdout.strip()

        return {"success": True, "url": pr_url, "error": None}

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "url": None,
            "error": f"Git command failed: {e.stderr.decode() if e.stderr else str(e)}",
        }
    except Exception as e:
        return {"success": False, "url": None, "error": str(e)}


def main():
    """CLI entry point for model deployment."""
    parser = argparse.ArgumentParser(
        description="Deploy trained neural network models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Deployment command")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload model to cloud storage")
    upload_parser.add_argument("--model", required=True, help="Path to ONNX model")
    upload_parser.add_argument("--model-id", required=True, help="Model identifier")
    upload_parser.add_argument("--version", default="1.0.0", help="Model version")
    upload_parser.add_argument("--bucket", required=True, help="Storage bucket")
    upload_parser.add_argument(
        "--backend", choices=["r2", "s3", "gcs", "local"], default="r2"
    )

    # Registry update command
    registry_parser = subparsers.add_parser(
        "update-registry", help="Update model registry files"
    )
    registry_parser.add_argument("--model-id", required=True, help="Model identifier")
    registry_parser.add_argument("--name", required=True, help="Model display name")
    registry_parser.add_argument("--url", required=True, help="Model download URL")
    registry_parser.add_argument(
        "--architecture", choices=["mlp", "cnn", "transformer"], required=True
    )
    registry_parser.add_argument(
        "--encoding",
        choices=["onehot-6x7x3", "bitboard", "flat-binary"],
        default="flat-binary",
    )
    registry_parser.add_argument("--expected-elo", type=int, required=True)
    registry_parser.add_argument("--size-bytes", type=int, required=True)
    registry_parser.add_argument("--version", default="1.0.0")
    registry_parser.add_argument("--experiment", help="Experiment directory for metadata")
    registry_parser.add_argument(
        "--makefour-root", default=".", help="Path to MakeFour repository"
    )

    # Full deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", help="Full deployment: upload + registry update + PR"
    )
    deploy_parser.add_argument("--model", required=True, help="Path to ONNX model")
    deploy_parser.add_argument("--model-id", required=True, help="Model identifier")
    deploy_parser.add_argument("--name", required=True, help="Model display name")
    deploy_parser.add_argument("--version", default="1.0.0", help="Model version")
    deploy_parser.add_argument(
        "--architecture", choices=["mlp", "cnn", "transformer"], required=True
    )
    deploy_parser.add_argument(
        "--encoding",
        choices=["onehot-6x7x3", "bitboard", "flat-binary"],
        default="flat-binary",
    )
    deploy_parser.add_argument("--expected-elo", type=int, required=True)
    deploy_parser.add_argument("--bucket", required=True, help="Storage bucket")
    deploy_parser.add_argument(
        "--backend", choices=["r2", "s3", "gcs", "local"], default="r2"
    )
    deploy_parser.add_argument("--experiment", help="Experiment directory for metadata")
    deploy_parser.add_argument(
        "--makefour-root", default=".", help="Path to MakeFour repository"
    )
    deploy_parser.add_argument(
        "--create-pr", action="store_true", help="Create a pull request"
    )
    deploy_parser.add_argument("--pr-branch", help="Branch name for PR")

    # Create PR command
    pr_parser = subparsers.add_parser("create-pr", help="Create PR for deployed model")
    pr_parser.add_argument("--model-id", required=True, help="Model identifier")
    pr_parser.add_argument("--branch", required=True, help="Branch name")
    pr_parser.add_argument(
        "--makefour-root", default=".", help="Path to MakeFour repository"
    )

    args = parser.parse_args()

    if args.command == "upload":
        from .upload import ModelUploadConfig, StorageBackend, upload_model

        config = ModelUploadConfig(
            model_path=Path(args.model),
            model_id=args.model_id,
            version=args.version,
            bucket=args.bucket,
            backend=StorageBackend(args.backend),
        )
        result = upload_model(config)
        if result.success:
            print(f"Upload successful: {result.url}")
            # Save result for registry update
            with open(f"{args.model_id}-upload.json", "w") as f:
                json.dump(
                    {
                        "url": result.url,
                        "size_bytes": result.size_bytes,
                        "md5_hash": result.md5_hash,
                    },
                    f,
                    indent=2,
                )
        else:
            print(f"Upload failed: {result.error}")
            sys.exit(1)

    elif args.command == "update-registry":
        training_games = 0
        training_epochs = 0

        if args.experiment:
            exp_metadata = load_experiment_metadata(Path(args.experiment))
            training_games = exp_metadata.get("training_games", 0)
            training_epochs = exp_metadata.get("training_epochs", 0)

        config = RegistryConfig(
            model_id=args.model_id,
            name=args.name,
            version=args.version,
            architecture=args.architecture,
            expected_elo=args.expected_elo,
            size_bytes=args.size_bytes,
            url=args.url,
            encoding=args.encoding,
            training_games=training_games,
            training_epochs=training_epochs,
            training_date=datetime.now().strftime("%Y-%m-%d"),
            client_registry_path=Path(args.makefour_root) / "src/ai/neural/loader.ts",
            server_registry_path=Path(args.makefour_root)
            / "functions/api/models/index.ts",
        )
        results = update_registry(config)
        print(json.dumps(results, indent=2, default=str))

    elif args.command == "deploy":
        training_games = 0
        training_epochs = 0

        if args.experiment:
            exp_metadata = load_experiment_metadata(Path(args.experiment))
            training_games = exp_metadata.get("training_games", 0)
            training_epochs = exp_metadata.get("training_epochs", 0)

        results = full_deploy(
            model_path=Path(args.model),
            model_id=args.model_id,
            name=args.name,
            version=args.version,
            architecture=args.architecture,
            encoding=args.encoding,
            expected_elo=args.expected_elo,
            bucket=args.bucket,
            backend=StorageBackend(args.backend),
            makefour_root=Path(args.makefour_root),
            training_games=training_games,
            training_epochs=training_epochs,
            create_pr=args.create_pr,
            pr_branch=args.pr_branch,
        )

        print("\nDeployment Results:")
        print(json.dumps(results, indent=2, default=str))

        if results["upload"] and not results["upload"]["success"]:
            sys.exit(1)

    elif args.command == "create-pr":
        result = create_deployment_pr(
            makefour_root=Path(args.makefour_root),
            model_id=args.model_id,
            branch=args.branch,
        )
        if result["success"]:
            print(f"PR created: {result['url']}")
        else:
            print(f"PR creation failed: {result['error']}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
