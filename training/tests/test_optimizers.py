"""Tests for optimizer and scheduler factories."""

import pytest
import torch

from src.training.optimizers import (
    WarmupCosineScheduler,
    create_optimizer,
    create_scheduler,
    get_current_lr,
    set_lr,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)


class TestCreateOptimizer:
    """Tests for create_optimizer function."""

    def test_create_adam(self):
        """Test creating Adam optimizer."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model.parameters(),
            optimizer_type="adam",
            learning_rate=0.001,
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_create_adamw(self):
        """Test creating AdamW optimizer."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model.parameters(),
            optimizer_type="adamw",
            learning_rate=0.01,
            weight_decay=0.1,
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["weight_decay"] == 0.1

    def test_create_sgd(self):
        """Test creating SGD optimizer."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model.parameters(),
            optimizer_type="sgd",
            learning_rate=0.1,
            momentum=0.9,
        )

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.1
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_invalid_optimizer_type(self):
        """Test that invalid optimizer type raises error."""
        model = SimpleModel()
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(model.parameters(), optimizer_type="invalid")


class TestCreateScheduler:
    """Tests for create_scheduler function."""

    def test_create_cosine_scheduler(self):
        """Test creating cosine annealing scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            epochs=100,
        )

        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_step_scheduler(self):
        """Test creating step scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="step",
            epochs=100,
            gamma=0.1,
        )

        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_multistep_scheduler(self):
        """Test creating multistep scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="multistep",
            epochs=100,
            milestones=[30, 60, 90],
            gamma=0.1,
        )

        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)

    def test_create_onecycle_scheduler(self):
        """Test creating OneCycle scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="onecycle",
            epochs=100,
            steps_per_epoch=100,
        )

        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def test_create_warmup_cosine_scheduler(self):
        """Test creating warmup cosine scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="warmup_cosine",
            epochs=100,
            warmup_epochs=10,
        )

        assert scheduler is not None
        assert isinstance(scheduler, WarmupCosineScheduler)

    def test_none_scheduler(self):
        """Test that 'none' returns None."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="none",
            epochs=100,
        )

        assert scheduler is None

    def test_invalid_scheduler_type(self):
        """Test that invalid scheduler type raises error."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(optimizer, scheduler_type="invalid", epochs=100)

    def test_onecycle_requires_steps_per_epoch(self):
        """Test that OneCycle scheduler requires steps_per_epoch."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        with pytest.raises(ValueError, match="steps_per_epoch is required"):
            create_scheduler(optimizer, scheduler_type="onecycle", epochs=100)


class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler."""

    def test_warmup_phase(self):
        """Test learning rate during warmup phase."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=10,
            total_epochs=100,
            min_lr=0.0001,
        )

        # At epoch 0, LR should be very small (1/10 of base LR)
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(0.001, rel=0.01)

        # Step through warmup
        for epoch in range(10):
            scheduler.step()

        # At epoch 10 (end of warmup), LR should be at base LR
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(0.01, rel=0.01)

    def test_cosine_decay_phase(self):
        """Test learning rate during cosine decay phase."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=0,  # No warmup
            total_epochs=100,
            min_lr=0.0001,
        )

        # At start, LR should be at base
        assert scheduler.get_lr()[0] == pytest.approx(0.01, rel=0.01)

        # Step to end
        for _ in range(100):
            scheduler.step()

        # At end, LR should be at min
        assert scheduler.get_lr()[0] == pytest.approx(0.0001, rel=0.01)

    def test_lr_decreases_monotonically_after_warmup(self):
        """Test that LR decreases monotonically after warmup."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=5,
            total_epochs=50,
            min_lr=0.0001,
        )

        # Skip warmup
        for _ in range(5):
            scheduler.step()

        prev_lr = scheduler.get_lr()[0]

        # Check LR decreases after warmup
        for _ in range(40):
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
            assert current_lr <= prev_lr
            prev_lr = current_lr


class TestLRUtilities:
    """Tests for learning rate utility functions."""

    def test_get_current_lr(self):
        """Test getting current learning rate."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        lr = get_current_lr(optimizer)
        assert lr == 0.01

    def test_set_lr(self):
        """Test setting learning rate."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        set_lr(optimizer, 0.001)

        assert get_current_lr(optimizer) == 0.001

    def test_set_lr_all_param_groups(self):
        """Test that set_lr updates all parameter groups."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(
            [
                {"params": model.fc.weight, "lr": 0.01},
                {"params": model.fc.bias, "lr": 0.001},
            ]
        )

        set_lr(optimizer, 0.0001)

        for group in optimizer.param_groups:
            assert group["lr"] == 0.0001
