"""Utility script for running a Composer experiment from a TOML file.

Usage:
    python scripts/run_composer.py path/to/experiment.toml [--plot]

The script loads the configuration, executes all pipelines defined in the
TOML using :class:`malthusjax.composer.Composer`, and prints a summary table.
If ``--plot`` is provided the convergence figure is displayed (requires
matplotlib).

This is intended as a convenient wrapper for ad‑hoc experiment runs;
see ``examples/_DEMO_COMPOSER`` for more complex notebooks.
"""

import argparse
import sys

from malthusjax.composer import Composer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a MalthusJAX Composer experiment defined in a TOML file."
    )
    parser.add_argument("config", help="Path to the experiment TOML file.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show convergence plot after the run (requires matplotlib).",
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Write JSON/CSV artifacts to the configured output directory.",
    )
    args = parser.parse_args()

    # load and execute the experiment
    try:
        result = Composer.from_toml(args.config)
    except Exception as exc:  # pragma: no cover - user script
        print(f"Failed to load/execute experiment: {exc}")
        sys.exit(1)

    # display high-level summary
    print("\n=== Experiment summary ===")
    try:
        summary = result.summary_table()
    except Exception:
        # Some result types may not implement summary_table
        summary = {}
    print(summary)

    # if we ran multiple pipelines, report any that produced no metrics
    if hasattr(result, "pipelines"):
        for name, exp in result.pipelines.items():
            if not exp.runs:
                print(f"WARNING: pipeline '{name}' produced no runs")
            else:
                # check if all runs have empty metrics
                if all(not r.metrics for r in exp.runs):
                    print(f"WARNING: pipeline '{name}' has runs but no metrics; check run statuses")
                    for r in exp.runs:
                        print(f"  seed={r.seed} status={r.status} error={r.error}")

    if args.write_artifacts:
        # ExperimentResult defines write_artifacts; ComparisonResult does not.
        if hasattr(result, "write_artifacts"):
            try:
                result.write_artifacts()
                print(f"Artifacts written to {result.output_dir}")
            except Exception as e:  # pragma: no cover
                print(f"Failed to write artifacts: {e}")
        else:
            # ComparisonResult: iterate over pipelines and call writer on each
            from malthusjax.benchmarking.io import write_summary_json, write_histories_csv

            for name, exp in result.pipelines.items():
                out = exp.metadata.get("output_dir")
                if not out:
                    print(f"No output_dir metadata for pipeline '{name}'; skipping artifact write")
                    continue
                try:
                    write_summary_json(exp, out)
                    write_histories_csv(exp, out)
                    print(f"Artifacts for pipeline '{name}' written to {out}")
                except Exception as e:  # pragma: no cover
                    print(f"Failed to write artifacts for '{name}': {e}")
            # note: shared initial population not written here

    if args.plot:
        try:
            result.plot_convergence()
        except ImportError:
            print("matplotlib is not installed; cannot plot.")


if __name__ == "__main__":
    main()
