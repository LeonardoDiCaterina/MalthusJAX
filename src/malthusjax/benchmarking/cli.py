"""Unified CLI interface for MalthusJAX benchmarking and analysis."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from malthusjax.composer import Composer
from malthusjax.composer.catalog import OperatorCatalog


def _dump_results(comparison, out_dir: Path, config_path: Path) -> None:
    """Helper to save results conforming to the strict CLI structure."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    shutil.copy2(config_path, meta_dir / "config_snapshot.toml")

    # Data
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    for pipe_name, exp_result in comparison.pipelines.items():
        pipe_dir = data_dir / f"pipeline_{pipe_name}"
        pipe_dir.mkdir(exist_ok=True)
        for run_result in exp_result.runs:
            seed = run_result.seed
            with open(pipe_dir / f"seed_{seed}.json", "w") as f:
                json.dump(run_result.to_dict(), f, indent=2)


def handle_run(args: argparse.Namespace) -> int:
    """Handle `mjax run`."""
    config_path = args.config
    print(f"Running experiment from {config_path}...")
    t0 = time.time()
    comparison = Composer.from_toml(config_path, shared_initial_population=True)
    dur = time.time() - t0

    out_dir = Path("results") / config_path.stem
    _dump_results(comparison, out_dir, config_path)

    print(f"Experiment complete in {dur:.2f}s.")
    print(f"Raw data saved to {out_dir}/data")
    return 0


def handle_parity(args: argparse.Namespace) -> int:
    """Handle `mjax parity`."""
    config_path = args.config
    print(f"Running statistical parity execution from {config_path}...")
    t0 = time.time()
    # parity implies enforcing a shared initial pop for exact alignment
    comparison = Composer.from_toml(config_path, shared_initial_population=True)
    dur = time.time() - t0

    out_dir = Path("results") / config_path.stem
    _dump_results(comparison, out_dir, config_path)

    print(f"Parity execution complete in {dur:.2f}s.")
    print(f"Raw data saved to {out_dir}/data")
    return 0


from malthusjax.benchmarking.results import (
    ComparisonResult,
    ExperimentResult,
    MetaComparison,
    RunResult,
)
from malthusjax.benchmarking.statistics import (
    ExpectedDirection,
    HypothesisKind,
    MultipleTestingPolicy,
    Sidedness,
    StatisticalComparator,
    StatisticalComparisonSpec,
    paired_dataset_from_comparison,
)


def _load_comparison(results_dir: Path) -> ComparisonResult:
    """Rebuild a ComparisonResult from the disk schema."""
    data_dir = results_dir / "data"
    pipelines = {}
    for pipe_dir in data_dir.glob("pipeline_*"):
        pipe_name = pipe_dir.name.replace("pipeline_", "")
        runs = []
        for seed_file in pipe_dir.glob("seed_*.json"):
            with open(seed_file, "r") as f:
                data = json.load(f)
                # handle if it was just dumped via run_result.to_dict()
                runs.append(RunResult.from_dict(data))
        pipelines[pipe_name] = ExperimentResult(name=pipe_name, runs=runs)
    return ComparisonResult(
        pipelines=pipelines, shared_config={}, initial_population=None, negate_map={}
    )


def handle_analyze(args: argparse.Namespace) -> int:
    """Handle `mjax analyze`."""
    results_dir = args.results_dir
    print(f"Analyzing results in {results_dir}...")
    comparison = _load_comparison(results_dir)

    pipe_names = list(comparison.pipelines.keys())

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    if len(pipe_names) == 2:
        left, right = pipe_names[0], pipe_names[1]
        print(f"Running statistical parity analysis for {left} vs {right}...")
        spec = StatisticalComparisonSpec(
            metric_name="best_fitness",
            hypothesis_kind=HypothesisKind("location_shift"),
            sidedness=Sidedness("two_sided"),
            expected_direction=ExpectedDirection("left_lt_right"),
            alpha=0.05,
            multiple_testing=MultipleTestingPolicy("none"),
        )
        dataset = paired_dataset_from_comparison(comparison, left, right, spec)
        comparator = StatisticalComparator()
        suite = comparator.compare_suite([dataset], spec)

        with open(analysis_dir / "parity_summary.json", "w") as f:
            json.dump(suite.to_dict(), f, indent=2)

        md_text = suite.to_markdown()
        with open(analysis_dir / "parity_summary.md", "w") as f:
            f.write(md_text)

        print("\n--- Parity Summary ---")
        print(md_text)
        print("----------------------\n")
        print("Analysis generated in analysis/")
    else:
        # Just standard mean/std dumps
        print("Saving standard mean/std dumps and unified tables...")
        for name, exp in comparison.pipelines.items():
            summary = exp.aggregated_summary()
            with open(analysis_dir / f"{name}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        try:
            import pandas as pd

            table = comparison.summary_table()
            formatted = {}
            for pipe, metrics in table.items():
                formatted[pipe] = {}
                for k, v in metrics.items():
                    if v.get("ci_margin", 0.0) > 0.0:
                        formatted[pipe][k] = f"{v['mean']:.4g} ± {v['ci_margin']:.4g}"
                    else:
                        formatted[pipe][k] = f"{v['mean']:.4g}"
            df = pd.DataFrame(formatted).T
            df.to_csv(analysis_dir / "comparison_table.csv")
            md_text = df.to_markdown()
            with open(analysis_dir / "comparison_table.md", "w") as f:
                f.write(md_text)

            print("\n--- Comparison Summary ---")
            print(md_text)
            print("--------------------------\n")

            with open(analysis_dir / "comparison_table.tex", "w") as f:
                f.write(comparison.summary_table(latex=True))
        except Exception as e:
            print(f"Could not generate unified tables: {e}")

    return 0


def handle_plot(args: argparse.Namespace) -> int:
    """Handle `mjax plot`."""
    results_dir = args.results_dir
    print(f"Plotting results for {results_dir}...")

    comparison = _load_comparison(results_dir)
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Try to generate the combined convergence plot
    try:
        comparison.plot_convergence(save_path=plot_dir / "convergence.png")
        print("Generated plots/convergence.png")
    except Exception as e:
        print(f"Could not generate convergence plot: {e}")

    # Try to generate the boxplot comparison
    try:
        comparison.plot_boxplots(save_path=plot_dir / "fitness_distribution.png")
        print("Generated plots/fitness_distribution.png")
    except Exception as e:
        print(f"Could not generate boxplots: {e}")

    # Try to generate the timings boxplot
    try:
        comparison.plot_boxplots(metric_key="duration_seconds", save_path=plot_dir / "timings.png")
        print("Generated plots/timings.png")
    except Exception as e:
        print(f"Could not generate timings boxplot: {e}")

    return 0


def handle_report(args: argparse.Namespace) -> int:
    """Handle `mjax report`."""
    print(f"Generating full report for {args.results_dir}")
    # TODO: Chain analyze and plot
    handle_analyze(args)
    handle_plot(args)
    return 0


def handle_aggregate(args: argparse.Namespace) -> int:
    """Handle `mjax aggregate`."""
    out_dir = args.out_dir
    results_dirs = args.results_dirs

    print(f"Aggregating {len(results_dirs)} experiments into {out_dir}...")

    comparisons = {}
    for d in results_dirs:
        print(f"  Loading {d}...")
        try:
            comp = _load_comparison(d)
            comparisons[d.name] = comp
        except Exception as e:
            print(f"  Failed to load {d}: {e}")

    if not comparisons:
        print("No valid experiments loaded.")
        return 1

    meta = MetaComparison(comparisons)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    print("Generating aggregate convergence grid...")
    meta.plot_convergence_grid(save_path=plot_dir / "convergence_grid.png")

    print("Generating aggregate boxplot grid...")
    meta.plot_boxplot_grid(save_path=plot_dir / "fitness_distribution_grid.png")

    print("Generating aggregate timings grid...")
    meta.plot_boxplot_grid(metric_key="duration_seconds", save_path=plot_dir / "timings_grid.png")

    print("Generating aggregate summary JSON...")
    summary = meta.summary_table()
    with open(out_dir / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Aggregate report complete. Results saved in {out_dir}")
    return 0


def handle_catalog(args: argparse.Namespace) -> int:
    """Handle `mjax catalog`."""
    catalog = OperatorCatalog()
    available = catalog.list_available()
    print("--- MalthusJAX Operator Catalog ---")
    for key in sorted(available):
        print(f"  - {key}")
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mjax", description="MalthusJAX Unified Benchmarking & Analysis CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. RUN
    parser_run = subparsers.add_parser("run", help="Run an experiment from a TOML config")
    parser_run.add_argument("config", type=Path, help="Path to experiment TOML config")
    parser_run.set_defaults(func=handle_run)

    # 2. PARITY
    parser_parity = subparsers.add_parser(
        "parity", help="Run seed-aligned statistical parity between pipelines"
    )
    parser_parity.add_argument("config", type=Path, help="Path to parity TOML config")
    parser_parity.set_defaults(func=handle_parity)

    # 3. ANALYZE
    parser_analyze = subparsers.add_parser(
        "analyze", help="Calculate statistical summaries from raw data"
    )
    parser_analyze.add_argument("results_dir", type=Path, help="Directory containing raw JSON data")
    parser_analyze.set_defaults(func=handle_analyze)

    # 4. PLOT
    parser_plot = subparsers.add_parser(
        "plot", help="Generate diagnostic plots from raw data and analysis"
    )
    parser_plot.add_argument("results_dir", type=Path, help="Directory containing raw JSON data")
    parser_plot.set_defaults(func=handle_plot)

    # 5. REPORT (Analyze + Plot)
    parser_report = subparsers.add_parser(
        "report", help="Generate both statistical summaries and diagnostic plots"
    )
    parser_report.add_argument("results_dir", type=Path, help="Directory containing raw JSON data")
    parser_report.set_defaults(func=handle_report)

    # 6. AGGREGATE
    parser_aggregate = subparsers.add_parser(
        "aggregate", help="Aggregate multiple experiments into a suite report"
    )
    parser_aggregate.add_argument(
        "--out_dir", type=Path, required=True, help="Output directory for the aggregate suite"
    )
    parser_aggregate.add_argument(
        "results_dirs", type=Path, nargs="+", help="One or more experiment result directories"
    )
    parser_aggregate.set_defaults(func=handle_aggregate)

    # 7. CATALOG
    parser_catalog = subparsers.add_parser("catalog", help="List registered framework operators")
    parser_catalog.set_defaults(func=handle_catalog)

    parsed = parser.parse_args(args)
    try:
        return parsed.func(parsed)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
