"""Tests for training callbacks."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.training.callbacks import (
    Callback,
    CallbackList,
    Checkpoint,
    ConsoleLogger,
    EarlyStopping,
    FileLogger,
    TrainingState,
    load_checkpoint,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def create_training_state(
    epoch: int = 0,
    train_loss: float = 1.0,
    val_loss: float | None = None,
    val_metrics: dict | None = None,
) -> TrainingState:
    """Create a TrainingState for testing."""
    return TrainingState(
        epoch=epoch,
        global_step=epoch * 100,
        train_loss=train_loss,
        val_loss=val_loss,
        val_metrics=val_metrics or {},
        learning_rate=0.001,
        epoch_time=1.0,
    )


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_training_state_creation(self):
        """Test creating a training state."""
        state = TrainingState(
            epoch=5,
            global_step=500,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=0.001,
            epoch_time=10.0,
        )

        assert state.epoch == 5
        assert state.global_step == 500
        assert state.train_loss == 0.5
        assert state.val_loss == 0.6


class TestCallbackList:
    """Tests for CallbackList."""

    def test_callback_list_calls_all_callbacks(self):
        """Test that CallbackList calls all registered callbacks."""
        call_counts = {"begin": 0, "end": 0}

        class CountingCallback(Callback):
            def on_train_begin(self, model, **kwargs):
                call_counts["begin"] += 1

            def on_train_end(self, model, state, **kwargs):
                call_counts["end"] += 1

        callbacks = CallbackList([CountingCallback(), CountingCallback()])
        model = SimpleModel()
        state = create_training_state()

        callbacks.on_train_begin(model)
        callbacks.on_train_end(model, state)

        assert call_counts["begin"] == 2
        assert call_counts["end"] == 2

    def test_callback_list_epoch_end_early_stop(self):
        """Test that CallbackList respects early stopping."""

        class StopCallback(Callback):
            def on_epoch_end(self, model, state, **kwargs):
                return False  # Signal to stop training

        class ContinueCallback(Callback):
            def on_epoch_end(self, model, state, **kwargs):
                return True

        callbacks = CallbackList([ContinueCallback(), StopCallback()])
        model = SimpleModel()
        state = create_training_state()

        should_continue = callbacks.on_epoch_end(model, state)

        assert should_continue is False


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_early_stopping_no_improvement(self):
        """Test early stopping triggers after patience epochs."""
        early_stop = EarlyStopping(patience=3, metric="val_loss", mode="min")
        model = SimpleModel()

        # No improvement - loss stays the same
        for epoch in range(3):
            state = create_training_state(epoch=epoch, val_loss=1.0)
            should_continue = early_stop.on_epoch_end(model, state)
            assert should_continue is True

        # Fourth epoch with no improvement should trigger stopping
        state = create_training_state(epoch=3, val_loss=1.0)
        should_continue = early_stop.on_epoch_end(model, state)
        assert should_continue is False

    def test_early_stopping_with_improvement(self):
        """Test early stopping resets counter on improvement."""
        early_stop = EarlyStopping(patience=2, metric="val_loss", mode="min")
        model = SimpleModel()

        # Improving loss
        for epoch, loss in enumerate([1.0, 0.9, 0.8, 0.7]):
            state = create_training_state(epoch=epoch, val_loss=loss)
            should_continue = early_stop.on_epoch_end(model, state)
            assert should_continue is True

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode (e.g., accuracy)."""
        early_stop = EarlyStopping(patience=2, metric="policy_acc", mode="max")
        model = SimpleModel()

        # Improving accuracy
        state = create_training_state(
            epoch=0, val_loss=1.0, val_metrics={"policy_acc": 0.5}
        )
        assert early_stop.on_epoch_end(model, state) is True

        state = create_training_state(
            epoch=1, val_loss=1.0, val_metrics={"policy_acc": 0.6}
        )
        assert early_stop.on_epoch_end(model, state) is True

        # Decreasing accuracy
        state = create_training_state(
            epoch=2, val_loss=1.0, val_metrics={"policy_acc": 0.5}
        )
        assert early_stop.on_epoch_end(model, state) is True

        state = create_training_state(
            epoch=3, val_loss=1.0, val_metrics={"policy_acc": 0.5}
        )
        assert early_stop.on_epoch_end(model, state) is False


class TestCheckpoint:
    """Tests for Checkpoint callback."""

    def test_checkpoint_saves_periodically(self):
        """Test that checkpoints are saved every N epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Checkpoint(checkpoint_dir=tmpdir, save_every=2)
            model = SimpleModel()

            for epoch in range(5):
                state = create_training_state(epoch=epoch, train_loss=1.0)
                checkpoint.on_epoch_end(model, state)

            # Should have checkpoints at epochs 2 and 4
            checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))
            assert len(checkpoints) == 2
            assert (Path(tmpdir) / "checkpoint_epoch_2.pt").exists()
            assert (Path(tmpdir) / "checkpoint_epoch_4.pt").exists()

    def test_checkpoint_saves_best_model(self):
        """Test that best model is saved separately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Checkpoint(
                checkpoint_dir=tmpdir,
                save_every=100,  # Don't save periodic
                save_best=True,
                metric="val_loss",
                mode="min",
            )
            model = SimpleModel()

            # Improving loss
            losses = [1.0, 0.8, 0.9, 0.7, 0.75]
            for epoch, loss in enumerate(losses):
                state = create_training_state(epoch=epoch, val_loss=loss)
                checkpoint.on_epoch_end(model, state)

            best_path = Path(tmpdir) / "best_model.pt"
            assert best_path.exists()

            # Best model should be from epoch 3 (loss=0.7)
            ckpt = torch.load(best_path, weights_only=False)
            assert ckpt["epoch"] == 3
            assert ckpt["val_loss"] == 0.7

    def test_checkpoint_saves_final_model(self):
        """Test that final model is saved on train end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Checkpoint(checkpoint_dir=tmpdir, save_every=100)
            model = SimpleModel()
            state = create_training_state(epoch=9, train_loss=0.5)

            checkpoint.on_train_end(model, state)

            final_path = Path(tmpdir) / "final_model.pt"
            assert final_path.exists()


class TestFileLogger:
    """Tests for FileLogger callback."""

    def test_file_logger_creates_log(self):
        """Test that file logger creates a JSON log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "training_log.json"
            logger = FileLogger(log_path)
            model = SimpleModel()

            for epoch in range(3):
                state = create_training_state(
                    epoch=epoch,
                    train_loss=1.0 - epoch * 0.1,
                    val_loss=1.1 - epoch * 0.1,
                )
                logger.on_epoch_end(model, state)

            assert log_path.exists()

            with open(log_path) as f:
                history = json.load(f)

            assert len(history) == 3
            assert history[0]["epoch"] == 1
            assert history[0]["train_loss"] == 1.0
            assert history[2]["epoch"] == 3


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_checkpoint_restores_model(self):
        """Test that load_checkpoint restores model weights."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        # Ensure models have different weights
        with torch.no_grad():
            model1.fc.weight.fill_(1.0)
            model2.fc.weight.fill_(2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model1.state_dict(),
                    "epoch": 5,
                    "train_loss": 0.5,
                },
                checkpoint_path,
            )

            ckpt = load_checkpoint(checkpoint_path, model2)

            assert torch.allclose(model1.fc.weight, model2.fc.weight)
            assert ckpt["epoch"] == 5

    def test_load_checkpoint_restores_optimizer(self):
        """Test that load_checkpoint restores optimizer state."""
        model = SimpleModel()
        optimizer1 = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer2 = torch.optim.Adam(model.parameters(), lr=0.01)

        # Do some optimization steps to change optimizer state
        x = torch.randn(5, 10)
        for _ in range(10):
            optimizer1.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer1.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer1.state_dict(),
                    "epoch": 5,
                },
                checkpoint_path,
            )

            load_checkpoint(checkpoint_path, model, optimizer2)

            # Optimizer states should match
            for key in optimizer1.state_dict()["state"]:
                state1 = optimizer1.state_dict()["state"][key]
                state2 = optimizer2.state_dict()["state"][key]
                for k in state1:
                    if isinstance(state1[k], torch.Tensor):
                        assert torch.allclose(state1[k], state2[k])
                    else:
                        assert state1[k] == state2[k]
