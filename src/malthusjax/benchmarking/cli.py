"""CLI interface for MalthusJAX benchmarking."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ..composer import Composer


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run MalthusJAX benchmarks with sensible defaults"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Random seeds to run (default: 1 2 3)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="cli_experiment",
        help="Experiment name (default: cli_experiment)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: results/{name})"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations (default: 10)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )

    parsed = parser.parse_args(args)

    if not parsed.quiet:
        print(f"Running experiment '{parsed.name}' with seeds {parsed.seeds}")

    try:
        composer = Composer.create_default()
        result = composer.quick_run(
            seeds=parsed.seeds,
            experiment_name=parsed.name,
            output_dir=parsed.output_dir,
            generations=parsed.generations,
        )

        if not parsed.quiet:
            print(f"Completed {len(result.runs)} runs")
            if "artifact_paths" in result.metadata:
                output_dir = Path(result.metadata["artifact_paths"]["summary_json"]).parent
                print(f"Results written to: {output_dir}")

            # Show quick summary
            agg = result.aggregated_summary()
            if "best_fitness" in agg:
                mean_fitness = agg["best_fitness"]["mean"]
                print(f"Mean best fitness: {mean_fitness:.4f}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
