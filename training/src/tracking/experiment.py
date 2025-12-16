"""
Experiment management for neural network training.

Provides the Experiment class for tracking training runs, metrics,
and configurations.
"""

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ExperimentStatus(str, Enum):
    """Status of an experiment."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    # Model configuration
    model_type: str
    model_params: dict[str, Any] = field(default_factory=dict)

    # Training configuration
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    optimizer: str = "adam"

    # Data configuration
    data_source: str = "self_play"
    num_games: int = 50000

    # Additional settings
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": {
                "type": self.model_type,
                **self.model_params,
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "optimizer": self.optimizer,
            },
            "data": {
                "source": self.data_source,
                "num_games": self.num_games,
            },
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        model = data.get("model", {})
        training = data.get("training", {})
        data_config = data.get("data", {})

        # Extract known fields, put rest in extra
        known_keys = {"model", "training", "data"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            model_type=model.get("type", "unknown"),
            model_params={k: v for k, v in model.items() if k != "type"},
            epochs=training.get("epochs", 100),
            batch_size=training.get("batch_size", 256),
            learning_rate=training.get("learning_rate", 1e-3),
            optimizer=training.get("optimizer", "adam"),
            data_source=data_config.get("source", "self_play"),
            num_games=data_config.get("num_games", 50000),
            extra=extra,
        )


class Experiment:
    """
    Manages a single training experiment.

    Tracks configuration, metrics, and artifacts for reproducibility.
    """

    def __init__(
        self,
        name: str,
        config: ExperimentConfig | dict[str, Any],
        experiments_dir: str | Path = "experiments",
        experiment_id: str | None = None,
    ):
        """
        Create a new experiment.

        Args:
            name: Human-readable experiment name (e.g., "mlp-tiny-baseline")
            config: Experiment configuration
            experiments_dir: Base directory for experiments
            experiment_id: Optional custom ID (auto-generated if not provided)
        """
        self.id = experiment_id or self._generate_id()
        self.name = name
        self.config = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_dict(config)
        self.status = ExperimentStatus.CREATED
        self.created_at = datetime.now()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None

        # Git information
        self.git_commit = self._get_git_commit()
        self.git_branch = self._get_git_branch()

        # Metrics storage
        self.metrics: list[dict[str, Any]] = []

        # Evaluation results
        self.eval_result: dict[str, Any] | None = None

        # Set up directory
        self.experiments_dir = Path(experiments_dir)
        self.dir = self.experiments_dir / f"{self.id}-{self.name}"

    def _generate_id(self) -> str:
        """Generate a unique experiment ID."""
        # Format: exp-XXX where XXX is a counter
        timestamp = int(time.time()) % 100000
        return f"exp-{timestamp:05d}"

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    def _get_git_branch(self) -> str | None:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    def start(self) -> None:
        """Mark experiment as started and create directory structure."""
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now()

        # Create experiment directory
        self.dir.mkdir(parents=True, exist_ok=True)

        # Save initial config
        self.save_config()

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name (e.g., "train_loss", "val_accuracy")
            value: Metric value
            step: Optional step number
            epoch: Optional epoch number
        """
        metric = {
            "name": name,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
        if step is not None:
            metric["step"] = step
        if epoch is not None:
            metric["epoch"] = epoch

        self.metrics.append(metric)

        # Append to metrics file (append-only log)
        metrics_file = self.dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metric) + "\n")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None, epoch: int | None = None) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
            epoch: Optional epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step=step, epoch=epoch)

    def log_eval(self, eval_result: dict[str, Any]) -> None:
        """
        Log evaluation results.

        Args:
            eval_result: Evaluation results (e.g., ELO rating, match results)
        """
        self.eval_result = eval_result
        eval_file = self.dir / "eval.json"
        with open(eval_file, "w") as f:
            json.dump(eval_result, f, indent=2)

    def log_model(self, model_path: str | Path, name: str = "model_best") -> None:
        """
        Save a model checkpoint.

        Args:
            model_path: Path to model file (e.g., .onnx, .pt)
            name: Name for saved model (without extension)
        """
        src = Path(model_path)
        if not src.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        dst = self.dir / f"{name}{src.suffix}"
        shutil.copy(src, dst)

    def save_config(self) -> None:
        """Save experiment configuration to YAML."""
        config_data = {
            "experiment": {
                "id": self.id,
                "name": self.name,
                "status": self.status.value,
                "created_at": self.created_at.isoformat(),
                "git_commit": self.git_commit,
                "git_branch": self.git_branch,
            },
            **self.config.to_dict(),
        }

        config_file = self.dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    def complete(self, success: bool = True) -> None:
        """
        Mark experiment as completed.

        Args:
            success: Whether experiment completed successfully
        """
        self.status = ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED
        self.completed_at = datetime.now()

        # Update config with final status
        self.save_config()

        # Generate summary
        self.save_summary()

    def save_summary(self) -> None:
        """Generate and save experiment summary as README."""
        duration = None
        if self.started_at and self.completed_at:
            duration = self.completed_at - self.started_at

        lines = [
            f"# Experiment: {self.name}",
            "",
            f"**ID**: {self.id}",
            f"**Status**: {self.status.value}",
            f"**Created**: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if self.git_commit:
            lines.append(f"**Git Commit**: {self.git_commit}")
        if self.git_branch:
            lines.append(f"**Git Branch**: {self.git_branch}")
        if duration:
            lines.append(f"**Duration**: {duration}")

        lines.extend([
            "",
            "## Configuration",
            "",
            f"- **Model**: {self.config.model_type}",
            f"- **Epochs**: {self.config.epochs}",
            f"- **Batch Size**: {self.config.batch_size}",
            f"- **Learning Rate**: {self.config.learning_rate}",
            f"- **Data Source**: {self.config.data_source}",
            f"- **Num Games**: {self.config.num_games}",
        ])

        # Add final metrics summary
        if self.metrics:
            lines.extend(["", "## Final Metrics", ""])
            # Get last value for each metric
            last_metrics: dict[str, float] = {}
            for m in self.metrics:
                last_metrics[m["name"]] = m["value"]
            for name, value in sorted(last_metrics.items()):
                lines.append(f"- **{name}**: {value:.6f}")

        # Add eval results
        if self.eval_result:
            lines.extend(["", "## Evaluation", ""])
            if "elo" in self.eval_result:
                lines.append(f"- **ELO Rating**: {self.eval_result['elo']}")
            if "confidence_interval" in self.eval_result:
                ci = self.eval_result["confidence_interval"]
                lines.append(f"- **95% CI**: [{ci[0]}, {ci[1]}]")

        readme = self.dir / "README.md"
        with open(readme, "w") as f:
            f.write("\n".join(lines) + "\n")

    def to_dict(self) -> dict[str, Any]:
        """Convert experiment to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "config": self.config.to_dict(),
            "eval_result": self.eval_result,
            "dir": str(self.dir),
        }

    @classmethod
    def load(cls, experiment_dir: str | Path) -> "Experiment":
        """
        Load an experiment from disk.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Loaded Experiment instance
        """
        exp_dir = Path(experiment_dir)
        config_file = exp_dir / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file) as f:
            data = yaml.safe_load(f)

        exp_meta = data.pop("experiment", {})

        # Create experiment
        experiment = cls(
            name=exp_meta.get("name", exp_dir.name),
            config=data,
            experiments_dir=exp_dir.parent,
            experiment_id=exp_meta.get("id"),
        )

        # Restore status and timestamps
        if "status" in exp_meta:
            experiment.status = ExperimentStatus(exp_meta["status"])
        if "created_at" in exp_meta:
            experiment.created_at = datetime.fromisoformat(exp_meta["created_at"])
        experiment.git_commit = exp_meta.get("git_commit")
        experiment.git_branch = exp_meta.get("git_branch")

        # Load metrics
        metrics_file = exp_dir / "metrics.jsonl"
        if metrics_file.exists():
            with open(metrics_file) as f:
                experiment.metrics = [json.loads(line) for line in f if line.strip()]

        # Load eval results
        eval_file = exp_dir / "eval.json"
        if eval_file.exists():
            with open(eval_file) as f:
                experiment.eval_result = json.load(f)

        return experiment

    def get_metric_history(self, name: str) -> list[tuple[int, float]]:
        """
        Get history of a specific metric.

        Args:
            name: Metric name

        Returns:
            List of (step, value) tuples
        """
        history = []
        for m in self.metrics:
            if m["name"] == name:
                step = m.get("step") or m.get("epoch") or len(history)
                history.append((step, m["value"]))
        return history

    def get_final_metrics(self) -> dict[str, float]:
        """Get the last recorded value for each metric."""
        final: dict[str, float] = {}
        for m in self.metrics:
            final[m["name"]] = m["value"]
        return final

    def get_param_count(self) -> int | None:
        """Get model parameter count if recorded."""
        return self.config.model_params.get("param_count")
