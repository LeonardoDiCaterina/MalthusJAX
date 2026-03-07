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
    parser.add_argument(
        "--trace",
        type=str,
        default="results/traces",
        metavar="DIR",
        help=(
            "Enable JAX profiler tracing for seed[0] of each pipeline. "
            "Outputs Perfetto-compatible traces to DIR/<pipeline_name>/. "
            "Default is results/traces."
        ),
    )
    args = parser.parse_args()

    # load and execute the experiment
    try:
        result = Composer.from_toml(args.config, trace_dir=args.trace)
    except Exception as exc:  # pragma: no cover - user script
        print(f"Failed to load/execute experiment: {exc}")
        sys.exit(1)

    if args.trace:
        print(f"\nJAX profiler traces written to: {args.trace}/")

    # display high-level summary
    print("\n=== Experiment summary ===")
    try:
        summary = result.summary_table()
    except Exception:
        # Some result types may not implement summary_table
        summary = {}
    print(summary)

    # if result is a ComparisonResult we can also pull raw aggregated data to
    # highlight the starting fitness when it exists.
    if hasattr(result, "pipelines"):
        for name, exp in result.pipelines.items():
            agg = exp.aggregated_summary()
            if "initial_fitness" in agg:
                print(f"pipeline '{name}' initial_fitness = {agg['initial_fitness']['mean']}")

    # print runtime/duration information
    def _print_runtime(res_obj, label=""):
        durations = [r.duration_seconds for r in res_obj.runs if r.duration_seconds is not None]
        if durations:
            avg = sum(durations) / len(durations)
            print(f"{label}average duration per run: {avg:.3f} s (n={len(durations)})")
        # also show aggregated timing metrics if present
        # "compile" is shown separately with a note: it is paid at most once
        # per process (JAX caches compiled kernels), so it inflates only the
        # first seed.  Subsequent seeds see near-zero compile time.
        timing_keys = set()
        for r in res_obj.runs:
            if r.timings:
                timing_keys.update(r.timings.keys())
        if timing_keys:
            ORDERED = ["initialization", "compile", "evolution"]
            ordered = [k for k in ORDERED if k in timing_keys]
            ordered += sorted(k for k in timing_keys if k not in ORDERED)
            print(f"{label}average timings (averages over all seeds):")
            for key in ordered:
                vals = [r.timings.get(key, 0.0) for r in res_obj.runs if r.timings]
                if vals:
                    first = res_obj.runs[0].timings.get(key, 0.0) if res_obj.runs[0].timings else 0.0
                    avg_v = sum(vals) / len(vals)
                    note = " ← includes 1st-compile" if key == "compile" and first > 0.05 else ""
                    print(f"  {key}: {avg_v:.3f} s  (seed[0]={first:.3f} s){note}")

    if hasattr(result, "pipelines"):
        # comparison result: print for each pipeline separately
        for name, exp in result.pipelines.items():
            print(f"\n-- Pipeline '{name}' runtime info --")
            _print_runtime(exp, label="  ")
    else:
        _print_runtime(result)

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
            # ComparisonResult: artifacts already written by each quick_run call.
            # Simply report the paths if available.
            for name, exp in result.pipelines.items():
                paths = exp.metadata.get("artifact_paths")
                if paths:
                    print(f"Pipeline '{name}' artifacts:")
                    for k, p in paths.items():
                        print(f"  {k}: {p}")
                else:
                    # fallback: maybe output_dir was stored
                    out = exp.metadata.get("output_dir")
                    if out:
                        print(f"Artifacts for pipeline '{name}' located at {out}")
                    else:
                        print(f"No artifact metadata for pipeline '{name}'")
            # note: shared initial population not written here

    if args.plot:
        try:
            result.plot_convergence()
        except ImportError:
            print("matplotlib is not installed; cannot plot.")


if __name__ == "__main__":
    main()
