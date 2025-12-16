#!/usr/bin/env python3
"""
CLI for experiment management.

Provides commands for listing, comparing, and visualizing experiments.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracking import ExperimentRegistry, ExperimentStatus


def cmd_list(args: argparse.Namespace) -> int:
    """List experiments with optional filters."""
    registry = ExperimentRegistry(args.experiments_dir)

    # Parse filters
    status = None
    if args.status:
        try:
            status = ExperimentStatus(args.status)
        except ValueError:
            print(f"Invalid status: {args.status}", file=sys.stderr)
            return 1

    experiments = registry.list(
        status=status,
        model_type=args.model_type,
        sort_by=args.sort_by,
        reverse=not args.ascending,
    )

    if not experiments:
        print("No experiments found.")
        return 0

    # Format output
    if args.json:
        data = [exp.to_dict() for exp in experiments]
        print(json.dumps(data, indent=2))
    else:
        # Table format
        print(f"{'ID':<15} {'Name':<25} {'Model':<15} {'Status':<12} {'Params':>10} {'ELO':>8}")
        print("-" * 90)
        for exp in experiments:
            params = f"{exp.param_count:,}" if exp.param_count else "N/A"
            elo = f"{exp.elo:.0f}" if exp.elo else "N/A"
            print(f"{exp.id:<15} {exp.name:<25} {exp.model_type:<15} {exp.status.value:<12} {params:>10} {elo:>8}")

        print(f"\nTotal: {len(experiments)} experiments")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show details of a specific experiment."""
    registry = ExperimentRegistry(args.experiments_dir)

    try:
        exp = registry.get(args.experiment_id)
    except KeyError:
        print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error loading experiment: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(exp.to_dict(), indent=2))
    else:
        print(f"Experiment: {exp.name}")
        print(f"ID: {exp.id}")
        print(f"Status: {exp.status.value}")
        print(f"Created: {exp.created_at}")
        print(f"Git Commit: {exp.git_commit or 'N/A'}")
        print(f"Git Branch: {exp.git_branch or 'N/A'}")
        print()
        print("Configuration:")
        print(f"  Model: {exp.config.model_type}")
        print(f"  Epochs: {exp.config.epochs}")
        print(f"  Batch Size: {exp.config.batch_size}")
        print(f"  Learning Rate: {exp.config.learning_rate}")
        print(f"  Data Source: {exp.config.data_source}")
        print(f"  Num Games: {exp.config.num_games}")
        print()

        final_metrics = exp.get_final_metrics()
        if final_metrics:
            print("Final Metrics:")
            for name, value in sorted(final_metrics.items()):
                print(f"  {name}: {value:.6f}")
            print()

        if exp.eval_result:
            print("Evaluation:")
            print(f"  ELO: {exp.eval_result.get('elo', 'N/A')}")
            if "confidence_interval" in exp.eval_result:
                ci = exp.eval_result["confidence_interval"]
                print(f"  95% CI: [{ci[0]}, {ci[1]}]")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare metrics across experiments."""
    registry = ExperimentRegistry(args.experiments_dir)

    metrics = args.metrics.split(",") if args.metrics else None
    results = registry.compare(args.experiment_ids, metrics=metrics)

    if not results:
        print("No experiments found.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Determine columns
        if results:
            all_keys = set()
            for r in results:
                all_keys.update(r.keys())
            # Order: id, name, model_type, param_count, elo, then metrics
            base_cols = ["id", "name", "model_type", "param_count", "elo"]
            metric_cols = sorted(all_keys - set(base_cols))
            columns = [c for c in base_cols if c in all_keys] + metric_cols

            # Print header
            header = " | ".join(f"{c:>12}" for c in columns)
            print(header)
            print("-" * len(header))

            # Print rows
            for r in results:
                values = []
                for c in columns:
                    v = r.get(c)
                    if v is None:
                        values.append("N/A")
                    elif isinstance(v, float):
                        values.append(f"{v:.4f}")
                    elif isinstance(v, int):
                        values.append(f"{v:,}")
                    else:
                        values.append(str(v)[:12])
                print(" | ".join(f"{v:>12}" for v in values))

    return 0


def cmd_plot_scaling(args: argparse.Namespace) -> int:
    """Generate scaling curve plot."""
    try:
        from tracking.visualization import plot_scaling_curve
    except ImportError as e:
        print(f"Error importing visualization: {e}", file=sys.stderr)
        return 1

    registry = ExperimentRegistry(args.experiments_dir)

    model_types = args.model_types.split(",") if args.model_types else None

    try:
        fig = plot_scaling_curve(
            registry,
            model_types=model_types,
            output_path=args.output,
            fit_power_law=not args.no_fit,
        )
        print(f"Plot saved to: {args.output}")
        if not args.output:
            import matplotlib.pyplot as plt
            plt.show()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_plot_learning(args: argparse.Namespace) -> int:
    """Generate learning curve plot."""
    try:
        from tracking.visualization import plot_learning_curve
    except ImportError as e:
        print(f"Error importing visualization: {e}", file=sys.stderr)
        return 1

    registry = ExperimentRegistry(args.experiments_dir)

    try:
        fig = plot_learning_curve(
            args.experiment_ids,
            registry,
            metric=args.metric,
            output_path=args.output,
        )
        print(f"Plot saved to: {args.output}" if args.output else "")
        if not args.output:
            import matplotlib.pyplot as plt
            plt.show()
    except (KeyError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export experiment data."""
    registry = ExperimentRegistry(args.experiments_dir)

    try:
        exp = registry.get(args.experiment_id)
    except KeyError:
        print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
        return 1

    data = exp.to_dict()
    data["metrics"] = exp.metrics

    output_path = Path(args.output) if args.output else Path(f"{exp.id}_export.json")

    if args.format == "json":
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    elif args.format == "csv":
        # Export metrics as CSV
        import csv
        output_path = output_path.with_suffix(".csv")
        with open(output_path, "w", newline="") as f:
            if exp.metrics:
                fieldnames = list(exp.metrics[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(exp.metrics)

    print(f"Exported to: {output_path}")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Rebuild experiment index from disk."""
    registry = ExperimentRegistry(args.experiments_dir)
    count = registry.rebuild_index()
    print(f"Indexed {count} experiments.")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments-dir", "-d",
        default="experiments",
        help="Experiments directory (default: experiments)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--model-type", "-m", help="Filter by model type")
    list_parser.add_argument("--sort-by", default="created_at", help="Sort by field")
    list_parser.add_argument("--ascending", "-a", action="store_true", help="Ascending order")
    list_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=cmd_list)

    # show command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("experiment_id", help="Experiment ID")
    show_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    show_parser.set_defaults(func=cmd_show)

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiment_ids", nargs="+", help="Experiment IDs to compare")
    compare_parser.add_argument("--metrics", "-m", help="Comma-separated metrics to include")
    compare_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    compare_parser.set_defaults(func=cmd_compare)

    # plot-scaling command
    scaling_parser = subparsers.add_parser("plot-scaling", help="Generate scaling curve plot")
    scaling_parser.add_argument("--output", "-o", help="Output file path")
    scaling_parser.add_argument("--model-types", help="Comma-separated model types to include")
    scaling_parser.add_argument("--no-fit", action="store_true", help="Don't fit power law")
    scaling_parser.set_defaults(func=cmd_plot_scaling)

    # plot-learning command
    learning_parser = subparsers.add_parser("plot-learning", help="Generate learning curve plot")
    learning_parser.add_argument("experiment_ids", nargs="+", help="Experiment IDs to plot")
    learning_parser.add_argument("--metric", "-m", default="train_loss", help="Metric to plot")
    learning_parser.add_argument("--output", "-o", help="Output file path")
    learning_parser.set_defaults(func=cmd_plot_learning)

    # export command
    export_parser = subparsers.add_parser("export", help="Export experiment data")
    export_parser.add_argument("experiment_id", help="Experiment ID")
    export_parser.add_argument("--format", "-f", choices=["json", "csv"], default="json")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.set_defaults(func=cmd_export)

    # rebuild command
    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild experiment index")
    rebuild_parser.set_defaults(func=cmd_rebuild)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
