"""
Tests for ONNX export pipeline.

Tests export, validation, and metadata functionality for Connect Four models.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Skip all tests if onnx/onnxruntime not available
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import create_model, list_models
from export import (
    export_to_onnx,
    validate_onnx_model,
    add_metadata,
    get_metadata,
    ExportConfig,
    ModelMetadata,
)
from export.onnx_export import get_input_shape_for_model
from export.validation import compare_outputs, validate_with_game_positions
from export.metadata import create_metadata_from_model, get_typed_metadata, print_metadata


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExportConfig(output_path="test.onnx")
        assert config.opset_version == 14
        assert config.optimize is True
        assert config.quantize is False
        assert config.dynamic_batch is True
        assert config.input_name == "board"
        assert config.output_names == ["policy", "value"]

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExportConfig(
            output_path="custom.onnx",
            opset_version=12,
            optimize=False,
            quantize=True,
        )
        assert config.opset_version == 12
        assert config.optimize is False
        assert config.quantize is True


class TestONNXExport:
    """Tests for ONNX export functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.parametrize("model_name", ["mlp-micro", "cnn-micro"])
    def test_export_basic(self, model_name, temp_dir):
        """Test basic export for different model types."""
        model = create_model(model_name)
        output_path = temp_dir / f"{model_name}.onnx"

        config = ExportConfig(
            output_path=output_path,
            optimize=False,  # Skip optimization for speed
        )

        result = export_to_onnx(model, config)

        assert result.output_path.exists()
        assert result.model_size_bytes > 0
        assert result.opset_version == 14

    def test_export_creates_directory(self, temp_dir):
        """Test that export creates output directory if needed."""
        model = create_model("mlp-micro")
        output_path = temp_dir / "subdir" / "model.onnx"

        config = ExportConfig(output_path=output_path, optimize=False)
        result = export_to_onnx(model, config)

        assert result.output_path.exists()
        assert result.output_path.parent.exists()

    def test_export_with_optimization(self, temp_dir):
        """Test export with ONNX optimization."""
        model = create_model("mlp-micro")
        output_path = temp_dir / "optimized.onnx"

        config = ExportConfig(output_path=output_path, optimize=True)
        result = export_to_onnx(model, config)

        assert result.output_path.exists()
        # Optimization may or may not succeed depending on environment
        # Just verify it doesn't crash

    @pytest.mark.parametrize("model_name", ["mlp-micro", "mlp-tiny", "cnn-micro", "cnn-tiny"])
    def test_export_all_small_models(self, model_name, temp_dir):
        """Test export for all small model variants."""
        model = create_model(model_name)
        output_path = temp_dir / f"{model_name}.onnx"

        config = ExportConfig(output_path=output_path, optimize=False)
        result = export_to_onnx(model, config)

        assert result.output_path.exists()

    def test_input_shape_mlp(self):
        """Test input shape detection for MLP models."""
        model = create_model("mlp-micro")
        shape = get_input_shape_for_model(model)
        assert shape == (85,)

    def test_input_shape_cnn(self):
        """Test input shape detection for CNN models."""
        model = create_model("cnn-micro")
        shape = get_input_shape_for_model(model)
        assert shape == (3, 6, 7)


class TestONNXValidation:
    """Tests for ONNX validation functionality."""

    @pytest.fixture
    def exported_mlp(self, tmp_path):
        """Create an exported MLP model for testing."""
        model = create_model("mlp-micro")
        output_path = tmp_path / "mlp.onnx"
        config = ExportConfig(output_path=output_path, optimize=False)
        export_to_onnx(model, config)
        return output_path, model

    @pytest.fixture
    def exported_cnn(self, tmp_path):
        """Create an exported CNN model for testing."""
        model = create_model("cnn-micro")
        output_path = tmp_path / "cnn.onnx"
        config = ExportConfig(output_path=output_path, optimize=False)
        export_to_onnx(model, config)
        return output_path, model

    def test_validate_mlp(self, exported_mlp):
        """Test validation passes for correctly exported MLP."""
        onnx_path, model = exported_mlp
        result = validate_onnx_model(onnx_path, model, input_shape=(85,), num_samples=10)

        assert result.valid
        assert result.policy_max_diff < 1e-4
        assert result.value_max_diff < 1e-4
        assert result.num_test_cases == 10

    def test_validate_cnn(self, exported_cnn):
        """Test validation passes for correctly exported CNN."""
        onnx_path, model = exported_cnn
        result = validate_onnx_model(onnx_path, model, input_shape=(3, 6, 7), num_samples=10)

        assert result.valid
        assert result.policy_max_diff < 1e-4
        assert result.value_max_diff < 1e-4

    def test_validate_nonexistent_file(self):
        """Test validation handles missing file gracefully."""
        model = create_model("mlp-micro")
        result = validate_onnx_model("/nonexistent/path.onnx", model, input_shape=(85,))

        assert not result.valid
        assert result.error_message is not None

    def test_compare_outputs(self, exported_mlp):
        """Test single output comparison."""
        import onnxruntime as ort

        onnx_path, model = exported_mlp
        model.eval()

        # Create test input
        test_input = torch.randn(1, 85)

        # Get PyTorch output
        with torch.no_grad():
            pt_output = model(test_input)

        # Get ONNX output
        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {"board": test_input.numpy()})

        matches, details = compare_outputs(pt_output, onnx_output)

        assert matches
        assert details["policy_match"]
        assert details["value_match"]

    def test_validate_with_game_positions(self, exported_mlp):
        """Test validation with realistic game positions."""
        from data import encode_flat_binary

        onnx_path, model = exported_mlp
        result = validate_with_game_positions(
            onnx_path, model, encode_flat_binary
        )

        assert result.valid
        assert result.num_test_cases > 0


class TestMetadata:
    """Tests for ONNX metadata functionality."""

    @pytest.fixture
    def model_with_export(self, tmp_path):
        """Create an exported model for metadata testing."""
        model = create_model("cnn-micro")
        output_path = tmp_path / "model.onnx"
        config = ExportConfig(output_path=output_path, optimize=False)
        export_to_onnx(model, config)
        return output_path, model

    def test_add_and_get_metadata(self, model_with_export):
        """Test adding and retrieving metadata."""
        onnx_path, _ = model_with_export

        metadata = {
            "model_name": "test-model",
            "architecture": "cnn",
            "param_count": 1234,
        }
        add_metadata(onnx_path, metadata)

        retrieved = get_metadata(onnx_path)

        assert retrieved["model_name"] == "test-model"
        assert retrieved["architecture"] == "cnn"
        assert retrieved["param_count"] == "1234"  # Stored as string

    def test_model_metadata_dataclass(self):
        """Test ModelMetadata dataclass."""
        metadata = ModelMetadata(
            model_name="cnn-tiny-v1",
            architecture="cnn",
            param_count=15000,
            input_encoding="onehot-6x7x3",
            input_shape="3,6,7",
            training_games=100000,
            training_epochs=50,
            estimated_elo=1250,
        )

        # Test to_dict
        d = metadata.to_dict()
        assert d["model_name"] == "cnn-tiny-v1"
        assert d["param_count"] == "15000"
        assert d["training_games"] == "100000"

        # Test from_dict round-trip
        restored = ModelMetadata.from_dict(d)
        assert restored.model_name == metadata.model_name
        assert restored.param_count == metadata.param_count
        assert restored.training_games == metadata.training_games

    def test_create_metadata_from_model(self):
        """Test automatic metadata creation from model."""
        model = create_model("cnn-tiny")
        metadata = create_metadata_from_model(
            model,
            "cnn-tiny-v1",
            training_games=50000,
        )

        assert metadata.model_name == "cnn-tiny-v1"
        assert metadata.param_count > 0
        assert metadata.training_games == 50000
        assert metadata.export_date is not None

    def test_add_typed_metadata(self, model_with_export):
        """Test adding ModelMetadata directly."""
        onnx_path, model = model_with_export

        metadata = create_metadata_from_model(model, "test-model-v1")
        add_metadata(onnx_path, metadata)

        retrieved = get_typed_metadata(onnx_path)
        assert retrieved is not None
        assert retrieved.model_name == "test-model-v1"

    def test_print_metadata(self, model_with_export):
        """Test metadata printing."""
        onnx_path, model = model_with_export

        metadata = create_metadata_from_model(
            model,
            "cnn-micro-v1",
            training_games=10000,
            estimated_elo=1100,
        )
        add_metadata(onnx_path, metadata)

        output = print_metadata(onnx_path)

        assert "cnn-micro-v1" in output
        assert "10000" in output
        assert "1100" in output


class TestONNXInference:
    """Tests for ONNX inference functionality."""

    @pytest.fixture
    def exported_models(self, tmp_path):
        """Create exported models for inference testing."""
        models = {}
        for name in ["mlp-micro", "cnn-micro"]:
            model = create_model(name)
            output_path = tmp_path / f"{name}.onnx"
            config = ExportConfig(output_path=output_path, optimize=False)
            export_to_onnx(model, config)
            models[name] = (output_path, model)
        return models

    def test_onnx_batch_inference(self, exported_models):
        """Test ONNX models work with batch inputs."""
        import onnxruntime as ort

        onnx_path, _ = exported_models["mlp-micro"]
        session = ort.InferenceSession(str(onnx_path))

        # Test different batch sizes
        for batch_size in [1, 4, 16]:
            input_data = np.random.randn(batch_size, 85).astype(np.float32)
            outputs = session.run(None, {"board": input_data})

            policy, value = outputs
            assert policy.shape == (batch_size, 7)
            assert value.shape == (batch_size, 1)

    def test_onnx_output_ranges(self, exported_models):
        """Test ONNX output values are in expected ranges."""
        import onnxruntime as ort

        onnx_path, _ = exported_models["mlp-micro"]
        session = ort.InferenceSession(str(onnx_path))

        input_data = np.random.randn(10, 85).astype(np.float32)
        outputs = session.run(None, {"board": input_data})

        policy, value = outputs

        # Value should be in [-1, 1] (tanh output)
        assert np.all(value >= -1.0)
        assert np.all(value <= 1.0)

        # Policy should be finite (logits)
        assert np.all(np.isfinite(policy))

    def test_onnx_deterministic(self, exported_models):
        """Test ONNX inference is deterministic."""
        import onnxruntime as ort

        onnx_path, _ = exported_models["cnn-micro"]
        session = ort.InferenceSession(str(onnx_path))

        input_data = np.random.randn(1, 3, 6, 7).astype(np.float32)

        # Run twice
        outputs1 = session.run(None, {"board": input_data})
        outputs2 = session.run(None, {"board": input_data})

        # Should be identical
        np.testing.assert_array_equal(outputs1[0], outputs2[0])
        np.testing.assert_array_equal(outputs1[1], outputs2[1])


class TestExportIntegration:
    """Integration tests for full export pipeline."""

    def test_full_pipeline_mlp(self, tmp_path):
        """Test complete export pipeline for MLP model."""
        from data import encode_flat_binary

        # Create and optionally train model
        model = create_model("mlp-tiny")

        # Export
        output_path = tmp_path / "mlp-tiny.onnx"
        config = ExportConfig(output_path=output_path, optimize=True)
        result = export_to_onnx(model, config)

        # Add metadata
        metadata = create_metadata_from_model(
            model,
            "mlp-tiny-test",
            training_games=1000,
            training_epochs=10,
        )
        add_metadata(output_path, metadata)

        # Validate with random inputs
        val_result = validate_onnx_model(output_path, model, input_shape=(85,))
        assert val_result.valid

        # Validate with game positions
        game_val = validate_with_game_positions(output_path, model, encode_flat_binary)
        assert game_val.valid

        # Verify metadata
        retrieved = get_metadata(output_path)
        assert retrieved["model_name"] == "mlp-tiny-test"

    def test_full_pipeline_cnn(self, tmp_path):
        """Test complete export pipeline for CNN model."""
        from data import encode_onehot_3d

        # Create model
        model = create_model("cnn-tiny")

        # Export
        output_path = tmp_path / "cnn-tiny.onnx"
        config = ExportConfig(output_path=output_path, optimize=True)
        result = export_to_onnx(model, config)

        # Validate
        val_result = validate_onnx_model(output_path, model, input_shape=(3, 6, 7))
        assert val_result.valid

        # Size check (CNN should be reasonably small)
        assert result.model_size_kb < 500  # Less than 500KB

    @pytest.mark.parametrize("model_name", list_models()[:6])  # Test first 6 models
    def test_export_all_models(self, model_name, tmp_path):
        """Test export works for all registered models."""
        model = create_model(model_name)
        output_path = tmp_path / f"{model_name}.onnx"

        config = ExportConfig(output_path=output_path, optimize=False)
        result = export_to_onnx(model, config)

        assert result.output_path.exists()

        # Validate
        input_shape = get_input_shape_for_model(model)
        val_result = validate_onnx_model(output_path, model, input_shape, num_samples=5)
        assert val_result.valid, f"Validation failed for {model_name}: {val_result.error_message}"
