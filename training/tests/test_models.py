"""Tests for neural network model architectures."""

import pytest
import torch

from src.models import (
    ConnectFourModel,
    TinyMLP,
    TinyCNN,
    MicroTransformer,
    TinyResNet,
    MODEL_REGISTRY,
    create_model,
    list_models,
    get_model_info,
)
from src.models.mlp import create_mlp_micro, create_mlp_tiny, create_mlp_small
from src.models.cnn import create_cnn_micro, create_cnn_tiny, create_cnn_small
from src.models.transformer import create_transformer_micro, create_transformer_tiny, create_transformer_small
from src.models.resnet import create_resnet_micro, create_resnet_tiny, create_resnet_small


class TestBaseModel:
    """Tests for base model interface."""

    def test_all_models_inherit_from_base(self):
        """All models should inherit from ConnectFourModel."""
        for name in list_models():
            model = create_model(name)
            assert isinstance(model, ConnectFourModel)
            assert isinstance(model, torch.nn.Module)

    def test_model_has_required_methods(self):
        """Models should have required interface methods."""
        model = create_mlp_tiny()
        assert hasattr(model, "forward")
        assert hasattr(model, "predict")
        assert hasattr(model, "param_count")
        assert hasattr(model, "architecture_string")
        assert hasattr(model, "get_config")
        assert hasattr(model, "summary")


class TestMLPModels:
    """Tests for MLP model variants."""

    @pytest.fixture
    def flat_input(self):
        """Create flat-binary encoded input."""
        return torch.randn(4, 85)  # batch of 4

    def test_mlp_micro_forward(self, flat_input):
        """MLP micro should produce correct output shapes."""
        model = create_mlp_micro()
        policy, value = model(flat_input)

        assert policy.shape == (4, 7), f"Expected (4, 7), got {policy.shape}"
        assert value.shape == (4, 1), f"Expected (4, 1), got {value.shape}"

    def test_mlp_tiny_forward(self, flat_input):
        """MLP tiny should produce correct output shapes."""
        model = create_mlp_tiny()
        policy, value = model(flat_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_mlp_small_forward(self, flat_input):
        """MLP small should produce correct output shapes."""
        model = create_mlp_small()
        policy, value = model(flat_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_mlp_predict(self, flat_input):
        """Predict should return softmax probabilities and tanh values."""
        model = create_mlp_tiny()
        policy, value = model.predict(flat_input)

        # Policy should sum to 1 (softmax)
        policy_sums = policy.sum(dim=1)
        assert torch.allclose(policy_sums, torch.ones(4)), "Policy should sum to 1"

        # Value should be in [-1, 1] (tanh)
        assert (value >= -1).all() and (value <= 1).all(), "Value should be in [-1, 1]"

    def test_mlp_param_count_increases(self):
        """Larger variants should have more parameters."""
        micro = create_mlp_micro()
        tiny = create_mlp_tiny()
        small = create_mlp_small()

        assert micro.param_count() < tiny.param_count() < small.param_count()

    def test_mlp_custom_config(self):
        """Should support custom hidden sizes."""
        model = TinyMLP(input_size=85, hidden_sizes=[256, 128, 64])
        x = torch.randn(2, 85)
        policy, value = model(x)

        assert policy.shape == (2, 7)
        assert value.shape == (2, 1)


class TestCNNModels:
    """Tests for CNN model variants."""

    @pytest.fixture
    def spatial_input(self):
        """Create one-hot encoded spatial input."""
        return torch.randn(4, 3, 6, 7)  # batch of 4

    @pytest.fixture
    def flat_input(self):
        """Create flattened one-hot input."""
        return torch.randn(4, 126)  # batch of 4

    def test_cnn_micro_forward(self, spatial_input):
        """CNN micro should produce correct output shapes."""
        model = create_cnn_micro()
        policy, value = model(spatial_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_cnn_tiny_forward(self, spatial_input):
        """CNN tiny should produce correct output shapes."""
        model = create_cnn_tiny()
        policy, value = model(spatial_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_cnn_small_forward(self, spatial_input):
        """CNN small should produce correct output shapes."""
        model = create_cnn_small()
        policy, value = model(spatial_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_cnn_flat_input(self, flat_input):
        """CNN should handle flattened input."""
        model = create_cnn_tiny()
        policy, value = model(flat_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_cnn_param_count_increases(self):
        """Larger variants should have more parameters."""
        micro = create_cnn_micro()
        tiny = create_cnn_tiny()
        small = create_cnn_small()

        assert micro.param_count() < tiny.param_count() < small.param_count()


class TestTransformerModels:
    """Tests for Transformer model variants."""

    @pytest.fixture
    def flat_input(self):
        """Create flat-binary encoded input."""
        return torch.randn(4, 85)

    def test_transformer_micro_forward(self, flat_input):
        """Transformer micro should produce correct output shapes."""
        model = create_transformer_micro()
        policy, value = model(flat_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_transformer_tiny_forward(self, flat_input):
        """Transformer tiny should produce correct output shapes."""
        model = create_transformer_tiny()
        policy, value = model(flat_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_transformer_small_forward(self, flat_input):
        """Transformer small should produce correct output shapes."""
        model = create_transformer_small()
        policy, value = model(flat_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_transformer_param_count_increases(self):
        """Larger variants should have more parameters."""
        micro = create_transformer_micro()
        tiny = create_transformer_tiny()
        small = create_transformer_small()

        assert micro.param_count() < tiny.param_count() < small.param_count()


class TestResNetModels:
    """Tests for ResNet model variants."""

    @pytest.fixture
    def spatial_input(self):
        """Create one-hot encoded spatial input."""
        return torch.randn(4, 3, 6, 7)

    def test_resnet_micro_forward(self, spatial_input):
        """ResNet micro should produce correct output shapes."""
        model = create_resnet_micro()
        policy, value = model(spatial_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_resnet_tiny_forward(self, spatial_input):
        """ResNet tiny should produce correct output shapes."""
        model = create_resnet_tiny()
        policy, value = model(spatial_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_resnet_small_forward(self, spatial_input):
        """ResNet small should produce correct output shapes."""
        model = create_resnet_small()
        policy, value = model(spatial_input)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_resnet_param_count_increases(self):
        """Larger variants should have more parameters."""
        micro = create_resnet_micro()
        tiny = create_resnet_tiny()
        small = create_resnet_small()

        assert micro.param_count() < tiny.param_count() < small.param_count()


class TestModelRegistry:
    """Tests for model registry functionality."""

    def test_list_models_returns_all(self):
        """list_models should return all registered models."""
        models = list_models()

        assert "mlp-micro" in models
        assert "mlp-tiny" in models
        assert "mlp-small" in models
        assert "cnn-micro" in models
        assert "cnn-tiny" in models
        assert "cnn-small" in models
        assert "transformer-micro" in models
        assert "transformer-tiny" in models
        assert "transformer-small" in models
        assert "resnet-micro" in models
        assert "resnet-tiny" in models
        assert "resnet-small" in models

    def test_create_model_all_variants(self):
        """create_model should work for all registered models."""
        for name in list_models():
            model = create_model(name)
            assert isinstance(model, ConnectFourModel)

    def test_create_model_unknown_raises(self):
        """create_model should raise for unknown models."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent-model")

    def test_get_model_info_all_variants(self):
        """get_model_info should work for all registered models."""
        for name in list_models():
            info = get_model_info(name)
            assert "description" in info
            assert "architecture" in info
            assert "params" in info
            assert "input_encoding" in info

    def test_get_model_info_unknown_raises(self):
        """get_model_info should raise for unknown models."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_info("nonexistent-model")


class TestModelGradients:
    """Tests for gradient flow through models."""

    @pytest.mark.parametrize("model_name", list_models())
    def test_gradient_flow(self, model_name):
        """All models should support gradient computation."""
        model = create_model(model_name)

        # Create appropriate input
        info = get_model_info(model_name)
        if info["input_encoding"] == "flat-binary":
            x = torch.randn(2, 85, requires_grad=True)
        else:
            x = torch.randn(2, 3, 6, 7, requires_grad=True)

        # Forward pass
        policy, value = model(x)

        # Backward pass
        loss = policy.sum() + value.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelConfig:
    """Tests for model configuration serialization."""

    @pytest.mark.parametrize("model_name", list_models())
    def test_get_config(self, model_name):
        """All models should return valid config."""
        model = create_model(model_name)
        config = model.get_config()

        assert "architecture" in config
        assert "input_size" in config
        assert "param_count" in config
        assert config["param_count"] == model.param_count()


class TestModelSummary:
    """Tests for model summary generation."""

    @pytest.mark.parametrize("model_name", list_models())
    def test_summary(self, model_name):
        """All models should generate summaries."""
        model = create_model(model_name)
        summary = model.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Parameters" in summary
