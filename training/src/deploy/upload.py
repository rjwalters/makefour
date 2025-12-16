"""
Model Upload Script

Uploads trained ONNX models to cloud storage (R2, S3, GCS).
Supports verification of upload integrity.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class StorageBackend(Enum):
    """Supported cloud storage backends."""

    R2 = "r2"  # Cloudflare R2
    S3 = "s3"  # AWS S3
    GCS = "gcs"  # Google Cloud Storage
    LOCAL = "local"  # Local filesystem (for testing)


@dataclass
class ModelUploadConfig:
    """Configuration for model upload."""

    model_path: Path
    model_id: str
    version: str
    bucket: str
    backend: StorageBackend = StorageBackend.R2
    prefix: str = "models"
    public: bool = True
    # Credentials (optional - can use environment variables)
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    endpoint_url: Optional[str] = None  # For R2


@dataclass
class UploadResult:
    """Result of a model upload."""

    success: bool
    url: str
    size_bytes: int
    md5_hash: str
    error: Optional[str] = None


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def upload_to_r2(config: ModelUploadConfig) -> UploadResult:
    """Upload model to Cloudflare R2."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        return UploadResult(
            success=False,
            url="",
            size_bytes=0,
            md5_hash="",
            error="boto3 not installed. Run: pip install boto3",
        )

    access_key = config.access_key_id or os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = config.secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY")
    endpoint = config.endpoint_url or os.environ.get("R2_ENDPOINT_URL")

    if not all([access_key, secret_key, endpoint]):
        return UploadResult(
            success=False,
            url="",
            size_bytes=0,
            md5_hash="",
            error="Missing R2 credentials. Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL",
        )

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
    )

    file_size = config.model_path.stat().st_size
    file_hash = compute_file_hash(config.model_path)
    object_key = f"{config.prefix}/{config.model_id}/{config.version}/model.onnx"

    try:
        extra_args = {}
        if config.public:
            extra_args["ACL"] = "public-read"
        extra_args["ContentType"] = "application/octet-stream"

        s3_client.upload_file(
            str(config.model_path),
            config.bucket,
            object_key,
            ExtraArgs=extra_args,
        )

        # Construct public URL
        # R2 public URLs use: https://<bucket>.<account_id>.r2.cloudflarestorage.com/<key>
        # Or custom domain: https://models.makefour.org/<key>
        public_url = os.environ.get(
            "R2_PUBLIC_URL", f"https://{config.bucket}.r2.dev"
        )
        url = f"{public_url}/{object_key}"

        return UploadResult(
            success=True,
            url=url,
            size_bytes=file_size,
            md5_hash=file_hash,
        )

    except Exception as e:
        return UploadResult(
            success=False,
            url="",
            size_bytes=file_size,
            md5_hash=file_hash,
            error=str(e),
        )


def upload_to_s3(config: ModelUploadConfig) -> UploadResult:
    """Upload model to AWS S3."""
    try:
        import boto3
    except ImportError:
        return UploadResult(
            success=False,
            url="",
            size_bytes=0,
            md5_hash="",
            error="boto3 not installed. Run: pip install boto3",
        )

    access_key = config.access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = config.secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([access_key, secret_key]):
        return UploadResult(
            success=False,
            url="",
            size_bytes=0,
            md5_hash="",
            error="Missing AWS credentials. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY",
        )

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    file_size = config.model_path.stat().st_size
    file_hash = compute_file_hash(config.model_path)
    object_key = f"{config.prefix}/{config.model_id}/{config.version}/model.onnx"

    try:
        extra_args = {"ContentType": "application/octet-stream"}
        if config.public:
            extra_args["ACL"] = "public-read"

        s3_client.upload_file(
            str(config.model_path),
            config.bucket,
            object_key,
            ExtraArgs=extra_args,
        )

        url = f"https://{config.bucket}.s3.amazonaws.com/{object_key}"

        return UploadResult(
            success=True,
            url=url,
            size_bytes=file_size,
            md5_hash=file_hash,
        )

    except Exception as e:
        return UploadResult(
            success=False,
            url="",
            size_bytes=file_size,
            md5_hash=file_hash,
            error=str(e),
        )


def upload_to_gcs(config: ModelUploadConfig) -> UploadResult:
    """Upload model to Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        return UploadResult(
            success=False,
            url="",
            size_bytes=0,
            md5_hash="",
            error="google-cloud-storage not installed. Run: pip install google-cloud-storage",
        )

    file_size = config.model_path.stat().st_size
    file_hash = compute_file_hash(config.model_path)
    object_key = f"{config.prefix}/{config.model_id}/{config.version}/model.onnx"

    try:
        client = storage.Client()
        bucket = client.bucket(config.bucket)
        blob = bucket.blob(object_key)

        blob.upload_from_filename(str(config.model_path))

        if config.public:
            blob.make_public()

        url = f"https://storage.googleapis.com/{config.bucket}/{object_key}"

        return UploadResult(
            success=True,
            url=url,
            size_bytes=file_size,
            md5_hash=file_hash,
        )

    except Exception as e:
        return UploadResult(
            success=False,
            url="",
            size_bytes=file_size,
            md5_hash=file_hash,
            error=str(e),
        )


def upload_to_local(config: ModelUploadConfig) -> UploadResult:
    """Copy model to local directory (for testing)."""
    import shutil

    file_size = config.model_path.stat().st_size
    file_hash = compute_file_hash(config.model_path)

    dest_dir = Path(config.bucket) / config.prefix / config.model_id / config.version
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "model.onnx"

    try:
        shutil.copy2(config.model_path, dest_path)
        return UploadResult(
            success=True,
            url=str(dest_path.absolute()),
            size_bytes=file_size,
            md5_hash=file_hash,
        )
    except Exception as e:
        return UploadResult(
            success=False,
            url="",
            size_bytes=file_size,
            md5_hash=file_hash,
            error=str(e),
        )


def upload_model(config: ModelUploadConfig) -> UploadResult:
    """
    Upload a model to the configured storage backend.

    Args:
        config: Upload configuration

    Returns:
        UploadResult with status, URL, and metadata
    """
    if not config.model_path.exists():
        return UploadResult(
            success=False,
            url="",
            size_bytes=0,
            md5_hash="",
            error=f"Model file not found: {config.model_path}",
        )

    if not config.model_path.suffix == ".onnx":
        return UploadResult(
            success=False,
            url="",
            size_bytes=0,
            md5_hash="",
            error="Model file must be an ONNX file (.onnx extension)",
        )

    upload_funcs = {
        StorageBackend.R2: upload_to_r2,
        StorageBackend.S3: upload_to_s3,
        StorageBackend.GCS: upload_to_gcs,
        StorageBackend.LOCAL: upload_to_local,
    }

    upload_func = upload_funcs.get(config.backend)
    if not upload_func:
        return UploadResult(
            success=False,
            url="",
            size_bytes=0,
            md5_hash="",
            error=f"Unsupported storage backend: {config.backend}",
        )

    return upload_func(config)


def main():
    """CLI entry point for model upload."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload trained model to cloud storage")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--version", default="1.0.0", help="Model version")
    parser.add_argument("--bucket", required=True, help="Storage bucket name")
    parser.add_argument(
        "--backend",
        choices=["r2", "s3", "gcs", "local"],
        default="r2",
        help="Storage backend",
    )
    parser.add_argument("--prefix", default="models", help="Object key prefix")
    parser.add_argument("--private", action="store_true", help="Make model private")

    args = parser.parse_args()

    config = ModelUploadConfig(
        model_path=Path(args.model),
        model_id=args.model_id,
        version=args.version,
        bucket=args.bucket,
        backend=StorageBackend(args.backend),
        prefix=args.prefix,
        public=not args.private,
    )

    print(f"Uploading {config.model_path} to {config.backend.value}://{config.bucket}...")
    result = upload_model(config)

    if result.success:
        print(f"Upload successful!")
        print(f"  URL: {result.url}")
        print(f"  Size: {result.size_bytes} bytes")
        print(f"  MD5: {result.md5_hash}")

        # Save result to JSON for registry update
        result_path = Path(f"{config.model_id}-upload-result.json")
        with open(result_path, "w") as f:
            json.dump(
                {
                    "url": result.url,
                    "size_bytes": result.size_bytes,
                    "md5_hash": result.md5_hash,
                    "model_id": config.model_id,
                    "version": config.version,
                },
                f,
                indent=2,
            )
        print(f"  Result saved to: {result_path}")
    else:
        print(f"Upload failed: {result.error}")
        exit(1)


if __name__ == "__main__":
    main()
