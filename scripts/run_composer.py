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
    print(result.summary_table())

    if args.write_artifacts:
        result.write_artifacts()
        print(f"Artifacts written to {result.output_dir}")

    if args.plot:
        try:
            result.plot_convergence()
        except ImportError:
            print("matplotlib is not installed; cannot plot.")


if __name__ == "__main__":
    main()
