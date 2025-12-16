"""
Experiment registry for indexing and searching experiments.

Provides the ExperimentRegistry class for managing multiple experiments.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .experiment import Experiment, ExperimentStatus


@dataclass
class ExperimentSummary:
    """Lightweight experiment summary for listings."""

    id: str
    name: str
    status: ExperimentStatus
    model_type: str
    param_count: int | None
    elo: float | None
    created_at: str
    dir: str

    @classmethod
    def from_experiment(cls, exp: Experiment) -> "ExperimentSummary":
        """Create summary from full experiment."""
        return cls(
            id=exp.id,
            name=exp.name,
            status=exp.status,
            model_type=exp.config.model_type,
            param_count=exp.get_param_count(),
            elo=exp.eval_result.get("elo") if exp.eval_result else None,
            created_at=exp.created_at.isoformat(),
            dir=str(exp.dir),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "model_type": self.model_type,
            "param_count": self.param_count,
            "elo": self.elo,
            "created_at": self.created_at,
            "dir": self.dir,
        }


class ExperimentRegistry:
    """
    Registry for indexing and searching experiments.

    Maintains an index file for quick lookups without loading all experiments.
    """

    def __init__(self, experiments_dir: str | Path = "experiments"):
        """
        Initialize the registry.

        Args:
            experiments_dir: Base directory for experiments
        """
        self.dir = Path(experiments_dir)
        self.index_path = self.dir / "index.json"
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load index from disk."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                data = json.load(f)
                self._index = data.get("experiments", {})

    def _save_index(self) -> None:
        """Save index to disk."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump({"experiments": self._index}, f, indent=2)

    def register(self, experiment: Experiment) -> None:
        """
        Register an experiment in the index.

        Args:
            experiment: Experiment to register
        """
        summary = ExperimentSummary.from_experiment(experiment)
        self._index[experiment.id] = summary.to_dict()
        self._save_index()

    def update(self, experiment: Experiment) -> None:
        """
        Update an experiment in the index.

        Args:
            experiment: Experiment to update
        """
        self.register(experiment)  # Same as register

    def unregister(self, experiment_id: str) -> None:
        """
        Remove an experiment from the index.

        Args:
            experiment_id: ID of experiment to remove
        """
        if experiment_id in self._index:
            del self._index[experiment_id]
            self._save_index()

    def get(self, experiment_id: str) -> Experiment:
        """
        Load a full experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Loaded Experiment

        Raises:
            KeyError: If experiment not found
        """
        if experiment_id not in self._index:
            raise KeyError(f"Experiment not found: {experiment_id}")

        exp_dir = self._index[experiment_id]["dir"]
        return Experiment.load(exp_dir)

    def get_summary(self, experiment_id: str) -> ExperimentSummary:
        """
        Get experiment summary without loading full experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            ExperimentSummary

        Raises:
            KeyError: If experiment not found
        """
        if experiment_id not in self._index:
            raise KeyError(f"Experiment not found: {experiment_id}")

        data = self._index[experiment_id]
        return ExperimentSummary(
            id=data["id"],
            name=data["name"],
            status=ExperimentStatus(data["status"]),
            model_type=data["model_type"],
            param_count=data.get("param_count"),
            elo=data.get("elo"),
            created_at=data["created_at"],
            dir=data["dir"],
        )

    def list(
        self,
        status: ExperimentStatus | None = None,
        model_type: str | None = None,
        min_elo: float | None = None,
        max_elo: float | None = None,
        sort_by: str = "created_at",
        reverse: bool = True,
    ) -> list[ExperimentSummary]:
        """
        List experiments with optional filters.

        Args:
            status: Filter by status
            model_type: Filter by model type
            min_elo: Filter by minimum ELO
            max_elo: Filter by maximum ELO
            sort_by: Field to sort by (created_at, elo, param_count, name)
            reverse: Reverse sort order (default: newest first)

        Returns:
            List of ExperimentSummary matching filters
        """
        results = []

        for data in self._index.values():
            # Apply filters
            if status and data["status"] != status.value:
                continue
            if model_type and data["model_type"] != model_type:
                continue
            if min_elo is not None and (data.get("elo") is None or data["elo"] < min_elo):
                continue
            if max_elo is not None and (data.get("elo") is None or data["elo"] > max_elo):
                continue

            summary = ExperimentSummary(
                id=data["id"],
                name=data["name"],
                status=ExperimentStatus(data["status"]),
                model_type=data["model_type"],
                param_count=data.get("param_count"),
                elo=data.get("elo"),
                created_at=data["created_at"],
                dir=data["dir"],
            )
            results.append(summary)

        # Sort results
        def get_sort_key(s: ExperimentSummary) -> Any:
            if sort_by == "elo":
                return s.elo or 0
            elif sort_by == "param_count":
                return s.param_count or 0
            elif sort_by == "name":
                return s.name
            else:  # created_at
                return s.created_at

        results.sort(key=get_sort_key, reverse=reverse)
        return results

    def list_ids(self) -> list[str]:
        """List all experiment IDs."""
        return list(self._index.keys())

    def exists(self, experiment_id: str) -> bool:
        """Check if experiment exists."""
        return experiment_id in self._index

    def count(self) -> int:
        """Count total experiments."""
        return len(self._index)

    def compare(
        self,
        experiment_ids: list[str],
        metrics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compare metrics across experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Optional list of metrics to include (all if None)

        Returns:
            List of dictionaries with experiment data and metrics
        """
        results = []

        for exp_id in experiment_ids:
            try:
                exp = self.get(exp_id)
            except (KeyError, FileNotFoundError):
                continue

            row = {
                "id": exp.id,
                "name": exp.name,
                "model_type": exp.config.model_type,
                "param_count": exp.get_param_count(),
                "elo": exp.eval_result.get("elo") if exp.eval_result else None,
            }

            # Add final metrics
            final_metrics = exp.get_final_metrics()
            if metrics:
                for m in metrics:
                    row[m] = final_metrics.get(m)
            else:
                row.update(final_metrics)

            results.append(row)

        return results

    def find_by_filter(self, filter_fn: Callable[[ExperimentSummary], bool]) -> list[ExperimentSummary]:
        """
        Find experiments matching a custom filter function.

        Args:
            filter_fn: Function that takes ExperimentSummary and returns bool

        Returns:
            List of matching ExperimentSummary
        """
        all_experiments = self.list()
        return [exp for exp in all_experiments if filter_fn(exp)]

    def rebuild_index(self) -> int:
        """
        Rebuild index from disk by scanning experiment directories.

        Returns:
            Number of experiments indexed
        """
        self._index = {}

        if not self.dir.exists():
            self._save_index()
            return 0

        count = 0
        for exp_dir in self.dir.iterdir():
            if not exp_dir.is_dir():
                continue
            if exp_dir.name == "index.json":
                continue

            config_file = exp_dir / "config.yaml"
            if not config_file.exists():
                continue

            try:
                exp = Experiment.load(exp_dir)
                self.register(exp)
                count += 1
            except Exception:
                # Skip invalid experiments
                continue

        return count

    def get_scaling_data(
        self,
        model_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get data for scaling curve analysis.

        Args:
            model_types: Optional filter for model types

        Returns:
            List of dicts with param_count, elo, model_type
        """
        data = []

        for exp_id in self.list_ids():
            try:
                exp = self.get(exp_id)
            except (KeyError, FileNotFoundError):
                continue

            if exp.status != ExperimentStatus.COMPLETED:
                continue
            if not exp.eval_result or "elo" not in exp.eval_result:
                continue

            param_count = exp.get_param_count()
            if param_count is None:
                continue

            if model_types and exp.config.model_type not in model_types:
                continue

            data.append({
                "id": exp.id,
                "name": exp.name,
                "model_type": exp.config.model_type,
                "param_count": param_count,
                "elo": exp.eval_result["elo"],
                "num_games": exp.config.num_games,
            })

        return data
