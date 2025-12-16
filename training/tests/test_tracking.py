"""
Tests for experiment tracking system.

Tests the Experiment, ExperimentRegistry, and related functionality.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.tracking.experiment import Experiment, ExperimentConfig, ExperimentStatus
from src.tracking.registry import ExperimentRegistry, ExperimentSummary


@pytest.fixture
def temp_experiments_dir():
    """Create a temporary directory for experiments."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Create a sample experiment config."""
    return ExperimentConfig(
        model_type="mlp-tiny",
        model_params={"hidden_sizes": [64, 32], "param_count": 5000},
        epochs=50,
        batch_size=128,
        learning_rate=0.001,
        data_source="self_play",
        num_games=10000,
    )


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_to_dict(self, sample_config):
        """Test conversion to dictionary."""
        data = sample_config.to_dict()

        assert data["model"]["type"] == "mlp-tiny"
        assert data["model"]["hidden_sizes"] == [64, 32]
        assert data["training"]["epochs"] == 50
        assert data["training"]["batch_size"] == 128
        assert data["data"]["source"] == "self_play"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "model": {
                "type": "cnn-small",
                "channels": [16, 32],
                "param_count": 10000,
            },
            "training": {
                "epochs": 100,
                "batch_size": 256,
                "learning_rate": 0.0005,
                "optimizer": "adamw",
            },
            "data": {
                "source": "supervised",
                "num_games": 50000,
            },
        }

        config = ExperimentConfig.from_dict(data)

        assert config.model_type == "cnn-small"
        assert config.model_params["channels"] == [16, 32]
        assert config.epochs == 100
        assert config.optimizer == "adamw"
        assert config.data_source == "supervised"

    def test_from_dict_with_defaults(self):
        """Test that missing fields get defaults."""
        data = {
            "model": {"type": "mlp"},
        }

        config = ExperimentConfig.from_dict(data)

        assert config.model_type == "mlp"
        assert config.epochs == 100  # default
        assert config.batch_size == 256  # default


class TestExperiment:
    """Tests for Experiment class."""

    def test_create_experiment(self, temp_experiments_dir, sample_config):
        """Test experiment creation."""
        exp = Experiment(
            name="test-experiment",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )

        assert exp.name == "test-experiment"
        assert exp.status == ExperimentStatus.CREATED
        assert exp.config.model_type == "mlp-tiny"
        assert exp.id.startswith("exp-")

    def test_experiment_start(self, temp_experiments_dir, sample_config):
        """Test starting an experiment."""
        exp = Experiment(
            name="test-start",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )

        exp.start()

        assert exp.status == ExperimentStatus.RUNNING
        assert exp.started_at is not None
        assert exp.dir.exists()
        assert (exp.dir / "config.yaml").exists()

    def test_log_metric(self, temp_experiments_dir, sample_config):
        """Test logging metrics."""
        exp = Experiment(
            name="test-metrics",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()

        exp.log_metric("train_loss", 0.5, step=0)
        exp.log_metric("train_loss", 0.3, step=1)
        exp.log_metric("val_accuracy", 0.8, epoch=0)

        assert len(exp.metrics) == 3
        assert exp.metrics[0]["name"] == "train_loss"
        assert exp.metrics[0]["value"] == 0.5
        assert exp.metrics[1]["step"] == 1

        # Check metrics file
        metrics_file = exp.dir / "metrics.jsonl"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_log_metrics_batch(self, temp_experiments_dir, sample_config):
        """Test logging multiple metrics at once."""
        exp = Experiment(
            name="test-batch",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()

        exp.log_metrics({
            "train_loss": 0.5,
            "val_loss": 0.6,
            "accuracy": 0.75,
        }, epoch=1)

        assert len(exp.metrics) == 3
        final = exp.get_final_metrics()
        assert final["train_loss"] == 0.5
        assert final["val_loss"] == 0.6
        assert final["accuracy"] == 0.75

    def test_log_eval(self, temp_experiments_dir, sample_config):
        """Test logging evaluation results."""
        exp = Experiment(
            name="test-eval",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()

        eval_result = {
            "elo": 1200,
            "confidence_interval": [1150, 1250],
            "games_played": 100,
        }
        exp.log_eval(eval_result)

        assert exp.eval_result == eval_result
        assert (exp.dir / "eval.json").exists()

    def test_experiment_complete(self, temp_experiments_dir, sample_config):
        """Test completing an experiment."""
        exp = Experiment(
            name="test-complete",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        exp.log_metric("loss", 0.1, step=100)
        exp.complete(success=True)

        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.completed_at is not None
        assert (exp.dir / "README.md").exists()

    def test_experiment_failed(self, temp_experiments_dir, sample_config):
        """Test marking experiment as failed."""
        exp = Experiment(
            name="test-failed",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        exp.complete(success=False)

        assert exp.status == ExperimentStatus.FAILED

    def test_get_metric_history(self, temp_experiments_dir, sample_config):
        """Test getting metric history."""
        exp = Experiment(
            name="test-history",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()

        for i in range(10):
            exp.log_metric("loss", 1.0 - i * 0.1, step=i)

        history = exp.get_metric_history("loss")

        assert len(history) == 10
        assert history[0] == (0, 1.0)
        assert history[9][0] == 9
        assert history[9][1] == pytest.approx(0.1)

    def test_to_dict(self, temp_experiments_dir, sample_config):
        """Test serialization to dict."""
        exp = Experiment(
            name="test-dict",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()

        data = exp.to_dict()

        assert data["name"] == "test-dict"
        assert data["status"] == "running"
        assert "config" in data
        assert data["config"]["model"]["type"] == "mlp-tiny"

    def test_load_experiment(self, temp_experiments_dir, sample_config):
        """Test loading experiment from disk."""
        # Create and save experiment
        exp = Experiment(
            name="test-load",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        exp.log_metric("loss", 0.5, step=1)
        exp.log_eval({"elo": 1100})
        exp.complete()

        # Load from disk
        loaded = Experiment.load(exp.dir)

        assert loaded.id == exp.id
        assert loaded.name == exp.name
        assert loaded.status == ExperimentStatus.COMPLETED
        assert len(loaded.metrics) == 1
        assert loaded.eval_result["elo"] == 1100

    def test_get_param_count(self, temp_experiments_dir):
        """Test getting parameter count."""
        config = ExperimentConfig(
            model_type="mlp",
            model_params={"param_count": 5000},
        )
        exp = Experiment(
            name="test-params",
            config=config,
            experiments_dir=temp_experiments_dir,
        )

        assert exp.get_param_count() == 5000


class TestExperimentRegistry:
    """Tests for ExperimentRegistry."""

    def test_register_experiment(self, temp_experiments_dir, sample_config):
        """Test registering an experiment."""
        registry = ExperimentRegistry(temp_experiments_dir)

        exp = Experiment(
            name="test-register",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        registry.register(exp)

        assert registry.exists(exp.id)
        assert registry.count() == 1

    def test_get_experiment(self, temp_experiments_dir, sample_config):
        """Test getting an experiment by ID."""
        registry = ExperimentRegistry(temp_experiments_dir)

        exp = Experiment(
            name="test-get",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        registry.register(exp)

        loaded = registry.get(exp.id)

        assert loaded.name == exp.name
        assert loaded.config.model_type == exp.config.model_type

    def test_get_summary(self, temp_experiments_dir, sample_config):
        """Test getting experiment summary."""
        registry = ExperimentRegistry(temp_experiments_dir)

        exp = Experiment(
            name="test-summary",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        registry.register(exp)

        summary = registry.get_summary(exp.id)

        assert isinstance(summary, ExperimentSummary)
        assert summary.id == exp.id
        assert summary.name == "test-summary"
        assert summary.model_type == "mlp-tiny"

    def test_list_experiments(self, temp_experiments_dir, sample_config):
        """Test listing experiments."""
        registry = ExperimentRegistry(temp_experiments_dir)

        # Create multiple experiments
        for i in range(3):
            exp = Experiment(
                name=f"test-list-{i}",
                config=sample_config,
                experiments_dir=temp_experiments_dir,
                experiment_id=f"exp-{i:05d}",
            )
            exp.start()
            registry.register(exp)

        experiments = registry.list_experiments()

        assert len(experiments) == 3

    def test_list_with_filter(self, temp_experiments_dir):
        """Test listing with filters."""
        registry = ExperimentRegistry(temp_experiments_dir)

        # Create experiments with different model types
        for i, model_type in enumerate(["mlp-tiny", "cnn-small", "mlp-tiny"]):
            config = ExperimentConfig(model_type=model_type)
            exp = Experiment(
                name=f"test-{model_type}-{i}",
                config=config,
                experiments_dir=temp_experiments_dir,
                experiment_id=f"exp-filter-{i:05d}",
            )
            exp.start()
            registry.register(exp)

        mlp_experiments = registry.list_experiments(model_type="mlp-tiny")

        assert len(mlp_experiments) == 2

    def test_compare_experiments(self, temp_experiments_dir, sample_config):
        """Test comparing experiments."""
        registry = ExperimentRegistry(temp_experiments_dir)

        exp_ids = []
        for i in range(3):
            exp = Experiment(
                name=f"test-compare-{i}",
                config=sample_config,
                experiments_dir=temp_experiments_dir,
            )
            exp.start()
            exp.log_metric("train_loss", 0.5 - i * 0.1, step=10)
            registry.register(exp)
            exp_ids.append(exp.id)

        results = registry.compare(exp_ids, metrics=["train_loss"])

        assert len(results) == 3
        assert all("train_loss" in r for r in results)

    def test_unregister_experiment(self, temp_experiments_dir, sample_config):
        """Test removing experiment from registry."""
        registry = ExperimentRegistry(temp_experiments_dir)

        exp = Experiment(
            name="test-unregister",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        registry.register(exp)

        assert registry.exists(exp.id)

        registry.unregister(exp.id)

        assert not registry.exists(exp.id)

    def test_rebuild_index(self, temp_experiments_dir, sample_config):
        """Test rebuilding index from disk."""
        # Create experiments directly on disk
        for i in range(3):
            exp = Experiment(
                name=f"test-rebuild-{i}",
                config=sample_config,
                experiments_dir=temp_experiments_dir,
                experiment_id=f"exp-rebuild-{i:05d}",
            )
            exp.start()
            exp.complete()

        # Create fresh registry and rebuild
        registry = ExperimentRegistry(temp_experiments_dir)
        count = registry.rebuild_index()

        assert count == 3
        assert registry.count() == 3

    def test_get_scaling_data(self, temp_experiments_dir):
        """Test getting data for scaling analysis."""
        registry = ExperimentRegistry(temp_experiments_dir)

        # Create experiments with varying param counts and ELOs
        for i, (params, elo) in enumerate([(1000, 900), (5000, 1100), (10000, 1200)]):
            config = ExperimentConfig(
                model_type="mlp-tiny",
                model_params={"param_count": params},
            )
            exp = Experiment(
                name=f"test-scaling-{i}",
                config=config,
                experiments_dir=temp_experiments_dir,
                experiment_id=f"exp-scaling-{i:05d}",
            )
            exp.start()
            exp.log_eval({"elo": elo})
            exp.complete()
            registry.register(exp)

        data = registry.get_scaling_data()

        assert len(data) == 3
        assert all("param_count" in d for d in data)
        assert all("elo" in d for d in data)

    def test_list_ids(self, temp_experiments_dir, sample_config):
        """Test listing experiment IDs."""
        registry = ExperimentRegistry(temp_experiments_dir)

        exp = Experiment(
            name="test-ids",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        registry.register(exp)

        ids = registry.list_ids()

        assert exp.id in ids

    def test_find_by_filter(self, temp_experiments_dir):
        """Test finding experiments with custom filter."""
        registry = ExperimentRegistry(temp_experiments_dir)

        # Create experiments
        for i, params in enumerate([1000, 5000, 10000]):
            config = ExperimentConfig(
                model_type="mlp",
                model_params={"param_count": params},
            )
            exp = Experiment(
                name=f"test-filter-{params}",
                config=config,
                experiments_dir=temp_experiments_dir,
                experiment_id=f"exp-byfilter-{i:05d}",
            )
            exp.start()
            registry.register(exp)

        # Find experiments with param_count > 3000
        results = registry.find_by_filter(
            lambda s: s.param_count is not None and s.param_count > 3000
        )

        assert len(results) == 2


class TestExperimentSummary:
    """Tests for ExperimentSummary."""

    def test_from_experiment(self, temp_experiments_dir, sample_config):
        """Test creating summary from experiment."""
        exp = Experiment(
            name="test-summary",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()
        exp.log_eval({"elo": 1100})

        summary = ExperimentSummary.from_experiment(exp)

        assert summary.id == exp.id
        assert summary.name == "test-summary"
        assert summary.model_type == "mlp-tiny"
        assert summary.param_count == 5000
        assert summary.elo == 1100

    def test_to_dict(self, temp_experiments_dir, sample_config):
        """Test summary serialization."""
        exp = Experiment(
            name="test-dict",
            config=sample_config,
            experiments_dir=temp_experiments_dir,
        )
        exp.start()

        summary = ExperimentSummary.from_experiment(exp)
        data = summary.to_dict()

        assert data["name"] == "test-dict"
        assert data["status"] == "running"
        assert data["model_type"] == "mlp-tiny"


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_experiment_lifecycle(self, temp_experiments_dir):
        """Test complete experiment lifecycle."""
        registry = ExperimentRegistry(temp_experiments_dir)

        # Create experiment
        config = ExperimentConfig(
            model_type="cnn-small",
            model_params={"channels": [16, 32], "param_count": 8000},
            epochs=10,
        )

        exp = Experiment(
            name="integration-test",
            config=config,
            experiments_dir=temp_experiments_dir,
        )

        # Start
        exp.start()
        registry.register(exp)

        # Log training
        for epoch in range(10):
            exp.log_metrics({
                "train_loss": 1.0 - epoch * 0.08,
                "val_loss": 1.1 - epoch * 0.07,
                "val_accuracy": 0.5 + epoch * 0.04,
            }, epoch=epoch)

        # Log evaluation
        exp.log_eval({
            "elo": 1150,
            "confidence_interval": [1100, 1200],
            "games_played": 50,
        })

        # Complete
        exp.complete()
        registry.update(exp)

        # Verify
        loaded = registry.get(exp.id)
        assert loaded.status == ExperimentStatus.COMPLETED
        assert loaded.eval_result["elo"] == 1150
        assert len(loaded.metrics) == 30  # 10 epochs * 3 metrics

        final_metrics = loaded.get_final_metrics()
        assert final_metrics["train_loss"] == pytest.approx(0.28, abs=0.01)
        assert final_metrics["val_accuracy"] == pytest.approx(0.86, abs=0.01)
