"""
Tests for the model deployment module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from training.src.deploy.upload import (
    ModelUploadConfig,
    StorageBackend,
    UploadResult,
    compute_file_hash,
    upload_model,
    upload_to_local,
)
from training.src.deploy.registry_update import (
    RegistryConfig,
    ModelMetadata,
    generate_model_metadata,
    format_typescript_metadata,
    update_typescript_registry,
    load_experiment_metadata,
)


class TestComputeFileHash:
    """Tests for file hash computation."""

    def test_compute_hash_for_file(self, tmp_path):
        """Test computing MD5 hash for a file."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        hash_result = compute_file_hash(test_file)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32  # MD5 hash is 32 hex characters

    def test_same_content_same_hash(self, tmp_path):
        """Test that identical files produce the same hash."""
        file1 = tmp_path / "file1.bin"
        file2 = tmp_path / "file2.bin"
        content = b"identical content"

        file1.write_bytes(content)
        file2.write_bytes(content)

        assert compute_file_hash(file1) == compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path):
        """Test that different files produce different hashes."""
        file1 = tmp_path / "file1.bin"
        file2 = tmp_path / "file2.bin"

        file1.write_bytes(b"content 1")
        file2.write_bytes(b"content 2")

        assert compute_file_hash(file1) != compute_file_hash(file2)


class TestUploadToLocal:
    """Tests for local storage upload."""

    def test_upload_creates_directory_structure(self, tmp_path):
        """Test that upload creates proper directory structure."""
        model_file = tmp_path / "source" / "model.onnx"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"fake onnx model")

        dest_bucket = tmp_path / "dest"

        config = ModelUploadConfig(
            model_path=model_file,
            model_id="test-model",
            version="1.0.0",
            bucket=str(dest_bucket),
            backend=StorageBackend.LOCAL,
        )

        result = upload_to_local(config)

        assert result.success
        expected_path = dest_bucket / "models" / "test-model" / "1.0.0" / "model.onnx"
        assert expected_path.exists()

    def test_upload_returns_correct_size(self, tmp_path):
        """Test that upload reports correct file size."""
        model_file = tmp_path / "model.onnx"
        content = b"x" * 1000
        model_file.write_bytes(content)

        config = ModelUploadConfig(
            model_path=model_file,
            model_id="test-model",
            version="1.0.0",
            bucket=str(tmp_path / "dest"),
            backend=StorageBackend.LOCAL,
        )

        result = upload_to_local(config)

        assert result.success
        assert result.size_bytes == 1000


class TestUploadModel:
    """Tests for the main upload_model function."""

    def test_upload_fails_for_missing_file(self):
        """Test that upload fails gracefully for missing file."""
        config = ModelUploadConfig(
            model_path=Path("/nonexistent/path/model.onnx"),
            model_id="test-model",
            version="1.0.0",
            bucket="test-bucket",
            backend=StorageBackend.LOCAL,
        )

        result = upload_model(config)

        assert not result.success
        assert "not found" in result.error.lower()

    def test_upload_fails_for_non_onnx_file(self, tmp_path):
        """Test that upload fails for non-ONNX files."""
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"pytorch model")

        config = ModelUploadConfig(
            model_path=model_file,
            model_id="test-model",
            version="1.0.0",
            bucket=str(tmp_path / "dest"),
            backend=StorageBackend.LOCAL,
        )

        result = upload_model(config)

        assert not result.success
        assert "onnx" in result.error.lower()


class TestGenerateModelMetadata:
    """Tests for model metadata generation."""

    def test_generates_basic_metadata(self):
        """Test basic metadata generation."""
        config = RegistryConfig(
            model_id="test-model",
            name="Test Model",
            version="1.0.0",
            architecture="cnn",
            expected_elo=1500,
            size_bytes=10000,
            url="https://models.example.com/test.onnx",
            encoding="flat-binary",
        )

        metadata = generate_model_metadata(config)

        assert metadata.id == "test-model"
        assert metadata.name == "Test Model"
        assert metadata.architecture == "cnn"
        assert metadata.expected_elo == 1500

    def test_includes_training_metadata(self):
        """Test that training metadata is included when provided."""
        config = RegistryConfig(
            model_id="test-model",
            name="Test Model",
            version="1.0.0",
            architecture="mlp",
            expected_elo=1400,
            size_bytes=5000,
            url="https://models.example.com/test.onnx",
            encoding="flat-binary",
            training_games=100000,
            training_epochs=50,
            training_date="2024-01-15",
        )

        metadata = generate_model_metadata(config)

        assert metadata.training is not None
        assert metadata.training["games"] == 100000
        assert metadata.training["epochs"] == 50
        assert metadata.training["date"] == "2024-01-15"

    def test_omits_training_when_no_games(self):
        """Test that training is omitted when no games specified."""
        config = RegistryConfig(
            model_id="test-model",
            name="Test Model",
            version="1.0.0",
            architecture="mlp",
            expected_elo=1400,
            size_bytes=5000,
            url="https://models.example.com/test.onnx",
            encoding="flat-binary",
            training_games=0,
        )

        metadata = generate_model_metadata(config)

        assert metadata.training is None


class TestFormatTypescriptMetadata:
    """Tests for TypeScript formatting."""

    def test_formats_basic_metadata(self):
        """Test basic TypeScript formatting."""
        metadata = ModelMetadata(
            id="test-model",
            name="Test Model",
            architecture="cnn",
            expected_elo=1500,
            size_bytes=10000,
            url="https://example.com/model.onnx",
            version="1.0.0",
            encoding="flat-binary",
        )

        formatted = format_typescript_metadata(metadata)

        assert "id: 'test-model'" in formatted
        assert "name: 'Test Model'" in formatted
        assert "architecture: 'cnn'" in formatted
        assert "expectedElo: 1500" in formatted

    def test_includes_training_block(self):
        """Test that training block is included."""
        metadata = ModelMetadata(
            id="test-model",
            name="Test Model",
            architecture="mlp",
            expected_elo=1400,
            size_bytes=5000,
            url="https://example.com/model.onnx",
            version="1.0.0",
            encoding="flat-binary",
            training={"games": 100000, "epochs": 50, "date": "2024-01-15"},
        )

        formatted = format_typescript_metadata(metadata)

        assert "training: {" in formatted
        assert "games: 100000" in formatted
        assert "epochs: 50" in formatted
        assert "date: '2024-01-15'" in formatted


class TestLoadExperimentMetadata:
    """Tests for loading experiment metadata."""

    def test_loads_from_config_json(self, tmp_path):
        """Test loading metadata from config.json."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        config = {
            "model": {"architecture": "cnn", "encoding": "onehot-6x7x3"},
            "training": {"epochs": 100},
        }
        (exp_dir / "config.json").write_text(json.dumps(config))

        metadata = load_experiment_metadata(exp_dir)

        assert metadata["architecture"] == "cnn"
        assert metadata["encoding"] == "onehot-6x7x3"
        assert metadata["training_epochs"] == 100

    def test_loads_from_eval_json(self, tmp_path):
        """Test loading metadata from eval.json."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        eval_data = {"estimated_elo": 1600, "games_played": 50000}
        (exp_dir / "eval.json").write_text(json.dumps(eval_data))

        metadata = load_experiment_metadata(exp_dir)

        assert metadata["expected_elo"] == 1600
        assert metadata["training_games"] == 50000

    def test_handles_missing_files(self, tmp_path):
        """Test handling of missing experiment files."""
        exp_dir = tmp_path / "empty_experiment"
        exp_dir.mkdir()

        metadata = load_experiment_metadata(exp_dir)

        assert metadata == {}


class TestUpdateTypescriptRegistry:
    """Tests for TypeScript registry update."""

    def test_fails_for_missing_file(self, tmp_path):
        """Test that update fails for missing file."""
        metadata = ModelMetadata(
            id="test-model",
            name="Test Model",
            architecture="mlp",
            expected_elo=1200,
            size_bytes=1000,
            url="https://example.com/model.onnx",
            version="1.0.0",
            encoding="flat-binary",
        )

        success, error = update_typescript_registry(
            tmp_path / "nonexistent.ts", metadata
        )

        assert not success
        assert "not found" in error.lower()

    def test_fails_for_duplicate_model(self, tmp_path):
        """Test that update fails if model already exists."""
        registry_file = tmp_path / "registry.ts"
        registry_file.write_text(
            """
const MODEL_REGISTRY: ModelMetadata[] = [
  {
    id: 'test-model',
    name: 'Existing Model',
  },
]
"""
        )

        metadata = ModelMetadata(
            id="test-model",
            name="Test Model",
            architecture="mlp",
            expected_elo=1200,
            size_bytes=1000,
            url="https://example.com/model.onnx",
            version="1.0.0",
            encoding="flat-binary",
        )

        success, error = update_typescript_registry(registry_file, metadata)

        assert not success
        assert "already exists" in error.lower()

    def test_inserts_new_model(self, tmp_path):
        """Test successful insertion of new model."""
        registry_file = tmp_path / "registry.ts"
        registry_file.write_text(
            """
const MODEL_REGISTRY: ModelMetadata[] = [
  {
    id: 'existing-model',
    name: 'Existing Model',
  },
  // Future models will be added here
]
"""
        )

        metadata = ModelMetadata(
            id="new-model",
            name="New Model",
            architecture="cnn",
            expected_elo=1400,
            size_bytes=5000,
            url="https://example.com/new-model.onnx",
            version="1.0.0",
            encoding="flat-binary",
        )

        success, error = update_typescript_registry(registry_file, metadata)

        assert success, f"Failed with error: {error}"

        content = registry_file.read_text()
        assert "id: 'new-model'" in content
        assert "name: 'New Model'" in content
