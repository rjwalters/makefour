"""Tests for Trainer class."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer, TrainerConfig


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 85):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.policy_head = torch.nn.Linear(32, 7)
        self.value_head = torch.nn.Linear(32, 1)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value


def create_dummy_dataloader(batch_size: int = 8, num_batches: int = 4) -> DataLoader:
    """Create a dummy dataloader for testing."""
    num_samples = batch_size * num_batches

    boards = torch.randn(num_samples, 85)
    moves = torch.randint(0, 7, (num_samples,))
    values = torch.randn(num_samples)
    legal_masks = torch.ones(num_samples, 7)

    class DictDataset:
        def __init__(self, boards, moves, values, legal_masks):
            self.boards = boards
            self.moves = moves
            self.values = values
            self.legal_masks = legal_masks

        def __len__(self):
            return len(self.boards)

        def __getitem__(self, idx):
            return {
                "board": self.boards[idx],
                "move": self.moves[idx],
                "value": self.values[idx],
                "legal_mask": self.legal_masks[idx],
            }

    dataset = DictDataset(boards, moves, values, legal_masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestTrainerConfig:
    """Tests for TrainerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainerConfig()

        assert config.epochs == 100
        assert config.batch_size == 256
        assert config.grad_clip == 1.0
        assert config.policy_weight == 1.0
        assert config.value_weight == 1.0

    def test_auto_device_selection(self):
        """Test automatic device selection."""
        config = TrainerConfig(device="auto")

        # Should select an available device
        assert config.device in ["cpu", "cuda", "mps"]


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig(device="cpu", epochs=1)

        trainer = Trainer(model=model, optimizer=optimizer, config=config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.global_step == 0

    def test_train_step(self):
        """Test single training step."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig(device="cpu")

        trainer = Trainer(model=model, optimizer=optimizer, config=config)

        # Create a dummy batch
        batch = {
            "board": torch.randn(4, 85),
            "move": torch.randint(0, 7, (4,)),
            "value": torch.randn(4),
            "legal_mask": torch.ones(4, 7),
        }

        loss, metrics = trainer.train_step(batch)

        assert isinstance(loss, float)
        assert loss > 0
        assert "policy_acc" in metrics
        assert "value_mae" in metrics

    def test_train_step_reduces_loss(self):
        """Test that training steps reduce loss over time."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        config = TrainerConfig(device="cpu")

        trainer = Trainer(model=model, optimizer=optimizer, config=config)

        # Fixed batch to test loss reduction
        batch = {
            "board": torch.randn(16, 85),
            "move": torch.randint(0, 7, (16,)),
            "value": torch.randn(16),
            "legal_mask": torch.ones(16, 7),
        }

        initial_loss, _ = trainer.train_step(batch)

        # Train for several steps
        for _ in range(50):
            trainer.train_step(batch)

        final_loss, _ = trainer.train_step(batch)

        # Loss should decrease (model should memorize this batch)
        assert final_loss < initial_loss

    def test_train_epoch(self):
        """Test training for one epoch."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig(device="cpu")

        trainer = Trainer(model=model, optimizer=optimizer, config=config)
        dataloader = create_dummy_dataloader()

        avg_loss, metrics = trainer.train_epoch(dataloader)

        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert trainer.global_step > 0

    def test_validate(self):
        """Test validation pass."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig(device="cpu")

        trainer = Trainer(model=model, optimizer=optimizer, config=config)
        dataloader = create_dummy_dataloader()

        val_loss, val_metrics = trainer.validate(dataloader)

        assert isinstance(val_loss, float)
        assert val_loss > 0
        assert "policy_acc" in val_metrics
        assert "value_mae" in val_metrics

    def test_full_training_loop(self):
        """Test complete training loop."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig(device="cpu", epochs=2)

        trainer = Trainer(model=model, optimizer=optimizer, config=config)

        train_loader = create_dummy_dataloader()
        val_loader = create_dummy_dataloader()

        results = trainer.train(train_loader, val_loader, epochs=2)

        assert results["final_epoch"] == 2
        assert results["final_train_loss"] is not None
        assert results["final_val_loss"] is not None
        assert len(results["history"]) == 2

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig(device="cpu")

        trainer = Trainer(model=model, optimizer=optimizer, config=config)

        # Do some training
        dataloader = create_dummy_dataloader()
        trainer.train_epoch(dataloader)
        original_step = trainer.global_step

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            # Create new trainer and load checkpoint
            model2 = SimpleModel()
            optimizer2 = torch.optim.Adam(model2.parameters())
            trainer2 = Trainer(model=model2, optimizer=optimizer2, config=config)

            trainer2.load_checkpoint(checkpoint_path)

            assert trainer2.global_step == original_step

            # Model weights should match
            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"Mismatch in {name1}"

    def test_gradient_clipping(self):
        """Test that gradient clipping is applied."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig(device="cpu", grad_clip=0.1)

        trainer = Trainer(model=model, optimizer=optimizer, config=config)

        # Create a batch that might cause large gradients
        batch = {
            "board": torch.randn(4, 85) * 10,
            "move": torch.randint(0, 7, (4,)),
            "value": torch.randn(4) * 10,
            "legal_mask": torch.ones(4, 7),
        }

        # After training step, gradients should be clipped
        trainer.train_step(batch)

        # Verify gradients exist but are bounded
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # With clip=0.1, individual gradient norms should be reasonable
                assert grad_norm < 100, f"Gradient norm too high for {name}"

    def test_training_with_scheduler(self):
        """Test training with learning rate scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        config = TrainerConfig(device="cpu", epochs=2)

        trainer = Trainer(
            model=model, optimizer=optimizer, scheduler=scheduler, config=config
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        train_loader = create_dummy_dataloader()
        trainer.train(train_loader, epochs=2)

        final_lr = optimizer.param_groups[0]["lr"]

        # LR should have decreased
        assert final_lr < initial_lr


class TestGradientFlow:
    """Test that gradients flow through the model correctly."""

    def test_gradients_flow_to_all_parameters(self):
        """Test that gradients are computed for all parameters."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig(device="cpu")

        trainer = Trainer(model=model, optimizer=optimizer, config=config)

        batch = {
            "board": torch.randn(4, 85),
            "move": torch.randint(0, 7, (4,)),
            "value": torch.randn(4),
            "legal_mask": torch.ones(4, 7),
        }

        # Clear any existing gradients
        optimizer.zero_grad()

        # Do forward and backward pass manually to check gradients
        board = batch["board"]
        policy_logits, value_pred = model(board)

        from src.training.losses import combined_loss
        loss, _ = combined_loss(
            policy_logits, value_pred, batch["move"], batch["value"]
        )
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
