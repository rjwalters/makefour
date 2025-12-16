"""
Visualization utilities for experiment analysis.

Provides plotting functions for learning curves, scaling analysis,
and architecture comparisons.
"""

from pathlib import Path
from typing import Any

# Note: matplotlib is an optional dependency for visualization
# Import is deferred to allow the module to be imported without matplotlib installed


def _check_matplotlib() -> None:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_learning_curve(
    experiment_ids: list[str],
    registry: Any,  # ExperimentRegistry
    metric: str = "train_loss",
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Any:
    """
    Plot learning curves for one or more experiments.

    Args:
        experiment_ids: List of experiment IDs to plot
        registry: ExperimentRegistry instance
        metric: Metric to plot (e.g., "train_loss", "val_accuracy")
        output_path: Optional path to save the plot
        title: Optional plot title

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_id in experiment_ids:
        try:
            exp = registry.get(exp_id)
        except (KeyError, FileNotFoundError):
            continue

        history = exp.get_metric_history(metric)
        if not history:
            continue

        steps, values = zip(*history)
        label = f"{exp.name} ({exp.id})"
        ax.plot(steps, values, label=label, marker="o", markersize=3)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Learning Curve: {metric}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_train_val_comparison(
    experiment_id: str,
    registry: Any,  # ExperimentRegistry
    train_metric: str = "train_loss",
    val_metric: str = "val_loss",
    output_path: str | Path | None = None,
) -> Any:
    """
    Plot training vs validation curves for a single experiment.

    Args:
        experiment_id: Experiment ID
        registry: ExperimentRegistry instance
        train_metric: Training metric name
        val_metric: Validation metric name
        output_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    exp = registry.get(experiment_id)

    fig, ax = plt.subplots(figsize=(10, 6))

    train_history = exp.get_metric_history(train_metric)
    val_history = exp.get_metric_history(val_metric)

    if train_history:
        steps, values = zip(*train_history)
        ax.plot(steps, values, label="Train", color="blue")

    if val_history:
        steps, values = zip(*val_history)
        ax.plot(steps, values, label="Validation", color="orange")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training vs Validation: {exp.name}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_scaling_curve(
    registry: Any,  # ExperimentRegistry
    model_types: list[str] | None = None,
    output_path: str | Path | None = None,
    fit_power_law: bool = True,
) -> Any:
    """
    Plot scaling curve: parameter count vs ELO rating.

    Args:
        registry: ExperimentRegistry instance
        model_types: Optional list of model types to include
        output_path: Optional path to save the plot
        fit_power_law: Whether to fit and plot power law curve

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    data = registry.get_scaling_data(model_types=model_types)
    if not data:
        raise ValueError("No completed experiments with ELO ratings found")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model type
    by_type: dict[str, list[dict]] = {}
    for d in data:
        mt = d["model_type"]
        if mt not in by_type:
            by_type[mt] = []
        by_type[mt].append(d)

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    for i, (model_type, experiments) in enumerate(sorted(by_type.items())):
        params = [e["param_count"] for e in experiments]
        elos = [e["elo"] for e in experiments]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.scatter(params, elos, label=model_type, color=color, marker=marker, s=80)

        # Add labels for points
        for e in experiments:
            ax.annotate(
                e["name"],
                (e["param_count"], e["elo"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.7,
            )

    # Fit power law if requested
    if fit_power_law and len(data) >= 3:
        try:
            all_params = np.array([d["param_count"] for d in data])
            all_elos = np.array([d["elo"] for d in data])

            # Fit: ELO = a * params^b + c
            # Use log-linear fit for simplicity
            log_params = np.log10(all_params)
            coeffs = np.polyfit(log_params, all_elos, 1)

            fit_x = np.logspace(np.log10(all_params.min()), np.log10(all_params.max()), 100)
            fit_y = coeffs[0] * np.log10(fit_x) + coeffs[1]

            ax.plot(
                fit_x, fit_y,
                "--", color="gray", alpha=0.5,
                label=f"Fit: ELO = {coeffs[0]:.1f}*log(params) + {coeffs[1]:.1f}",
            )
        except Exception:
            pass  # Skip fitting if it fails

    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("ELO Rating")
    ax.set_title("Scaling Curve: Model Size vs Performance")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_architecture_comparison(
    registry: Any,  # ExperimentRegistry
    param_budget: int | None = None,
    tolerance: float = 0.5,
    output_path: str | Path | None = None,
) -> Any:
    """
    Compare architectures at similar parameter budgets.

    Args:
        registry: ExperimentRegistry instance
        param_budget: Target parameter count (optional, uses median if not specified)
        tolerance: Allowed deviation from param_budget (e.g., 0.5 = 50%)
        output_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    data = registry.get_scaling_data()
    if not data:
        raise ValueError("No completed experiments with ELO ratings found")

    # Determine param budget
    all_params = [d["param_count"] for d in data]
    if param_budget is None:
        param_budget = int(np.median(all_params))

    # Filter to experiments near the budget
    min_params = param_budget * (1 - tolerance)
    max_params = param_budget * (1 + tolerance)

    filtered = [d for d in data if min_params <= d["param_count"] <= max_params]
    if not filtered:
        raise ValueError(f"No experiments found near {param_budget} params")

    # Group by architecture
    by_arch: dict[str, list[dict]] = {}
    for d in filtered:
        arch = d["model_type"]
        if arch not in by_arch:
            by_arch[arch] = []
        by_arch[arch].append(d)

    fig, ax = plt.subplots(figsize=(10, 6))

    architectures = list(sorted(by_arch.keys()))
    x = np.arange(len(architectures))

    # Get best ELO for each architecture
    best_elos = []
    for arch in architectures:
        elos = [d["elo"] for d in by_arch[arch]]
        best_elos.append(max(elos))

    bars = ax.bar(x, best_elos, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(architectures))))

    ax.set_xticks(x)
    ax.set_xticklabels(architectures, rotation=45, ha="right")
    ax.set_ylabel("ELO Rating")
    ax.set_title(f"Architecture Comparison (~{param_budget:,} params)")

    # Add value labels on bars
    for bar, elo in zip(bars, best_elos):
        ax.annotate(
            f"{elo:.0f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_distribution(
    experiment_ids: list[str],
    registry: Any,  # ExperimentRegistry
    metric: str = "train_loss",
    output_path: str | Path | None = None,
) -> Any:
    """
    Plot distribution of final metric values across experiments.

    Args:
        experiment_ids: List of experiment IDs
        registry: ExperimentRegistry instance
        metric: Metric to analyze
        output_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    values = []
    labels = []

    for exp_id in experiment_ids:
        try:
            exp = registry.get(exp_id)
            final_metrics = exp.get_final_metrics()
            if metric in final_metrics:
                values.append(final_metrics[metric])
                labels.append(exp.name)
        except (KeyError, FileNotFoundError):
            continue

    if not values:
        raise ValueError(f"No experiments found with metric: {metric}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(range(len(values)), values, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Final {metric} by Experiment")
    ax.invert_yaxis()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def generate_experiment_report(
    experiment_id: str,
    registry: Any,  # ExperimentRegistry
    output_dir: str | Path,
) -> Path:
    """
    Generate a full report for an experiment with all plots.

    Args:
        experiment_id: Experiment ID
        registry: ExperimentRegistry instance
        output_dir: Directory to save report files

    Returns:
        Path to generated report directory
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    exp = registry.get(experiment_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate learning curve
    history = exp.get_metric_history("train_loss")
    if history:
        fig, ax = plt.subplots(figsize=(10, 6))
        steps, values = zip(*history)
        ax.plot(steps, values)
        ax.set_xlabel("Step")
        ax.set_ylabel("Training Loss")
        ax.set_title(f"Learning Curve: {exp.name}")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "learning_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Generate train vs val comparison if both exist
    train_history = exp.get_metric_history("train_loss")
    val_history = exp.get_metric_history("val_loss")
    if train_history and val_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        steps, values = zip(*train_history)
        ax.plot(steps, values, label="Train")
        steps, values = zip(*val_history)
        ax.plot(steps, values, label="Validation")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Train vs Validation: {exp.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "train_val.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Generate summary text
    summary = [
        f"# Experiment Report: {exp.name}",
        "",
        f"**ID**: {exp.id}",
        f"**Model**: {exp.config.model_type}",
        f"**Status**: {exp.status.value}",
        "",
        "## Configuration",
        f"- Epochs: {exp.config.epochs}",
        f"- Batch Size: {exp.config.batch_size}",
        f"- Learning Rate: {exp.config.learning_rate}",
        "",
        "## Final Metrics",
    ]

    for name, value in exp.get_final_metrics().items():
        summary.append(f"- {name}: {value:.6f}")

    if exp.eval_result:
        summary.extend([
            "",
            "## Evaluation Results",
            f"- ELO: {exp.eval_result.get('elo', 'N/A')}",
        ])

    summary_path = output_dir / "report.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary) + "\n")

    return output_dir
