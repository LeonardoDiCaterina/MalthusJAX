"""Unified CLI interface for MalthusJAX benchmarking and analysis."""

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from malthusjax.benchmarking.io import write_experiment_artifacts
from malthusjax.composer import Composer
from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.config import load_experiment_config
from malthusjax.core.diagnostics import print_environment_diagnostics, uninstall_crash_handler
from malthusjax.core.logger import configure_logging, get_logger

logger = get_logger("benchmarking.cli")


def handle_check_env(args: argparse.Namespace) -> int:
    """Handle `mjax check-env`."""
    print_environment_diagnostics()
    return 0


def _dump_results(comparison: Any, out_dir: Path, config_path: Path) -> None:
    """Helper to save results conforming to the strict CLI structure."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    shutil.copy2(config_path, meta_dir / "config_snapshot.toml")

    # Data
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    manifest: dict[str, Any] = {"schema_version": 1, "pipelines": {}}
    history_rows = []
    csv_enabled = False
    for name, experiment in comparison.pipelines.items():
        _validate_pipeline_name(name)
        pipe_dir = data_dir / f"pipeline_{name}"
        pipe_dir.mkdir()
        records = {}
        for run in experiment.runs:
            for artifact_name in ("archive", "pareto"):
                if artifact_name in run.artifacts:
                    run.artifacts[artifact_name] = (
                        Path(run.artifacts[artifact_name]).relative_to(out_dir).as_posix()
                    )
            path = pipe_dir / f"seed_{run.seed}.json"
            with path.open("x") as stream:
                json.dump(run.to_dict(), stream, indent=2, allow_nan=False)
            records[str(run.seed)] = path.relative_to(out_dir).as_posix()
        options = experiment.metadata.get("output_options", {})
        write_history = options.get("write_history_csv", True)
        csv_enabled |= write_history
        manifest["pipelines"][name] = {
            "records": records,
            "output_options": options,
            "maximize": comparison.negate_map.get(name, False),
        }
        write_experiment_artifacts(
            experiment, out_dir / "exports" / f"pipeline_{name}", write_history_csv=write_history
        )
        if write_history:
            history_rows.extend({**row, "pipeline": name} for row in experiment.combined_history())

    if csv_enabled:
        fields = {"pipeline", "seed"}
        fields.update(key for row in history_rows for key in row)
        with (out_dir / "histories_combined.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["pipeline", "seed", *sorted(fields - {"pipeline", "seed"})]
            )
            writer.writeheader()
            writer.writerows(history_rows)
    (meta_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def _validate_pipeline_name(name: str) -> None:
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid pipeline name for result export: {name!r}")


def _run_comparison(args: argparse.Namespace) -> int:
    config_path = args.config
    config_bytes = config_path.read_bytes()
    config = load_experiment_config(str(config_path), contents=config_bytes)
    for name in config.pipelines:
        _validate_pipeline_name(name)
    shared = config.meta["shared"]
    configured_dir = (
        getattr(args, "output_dir", None)
        or shared.get("output_dir")
        or config.meta.get("output_dir")
    )
    out_dir = Path(configured_dir or Path("results") / config_path.stem).absolute()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Reserve the destination before optimization. The staged directory is on
    # the same filesystem so publishing it is one rename.
    lock = out_dir.with_name(f".{out_dir.name}.lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"Result destination is reserved: {out_dir}. Choose a fresh --output-dir."
        ) from exc

    try:
        if out_dir.exists() or out_dir.is_symlink():
            raise ValueError(
                f"Result directory already exists: {out_dir}. Choose a fresh --output-dir."
            )
        with tempfile.TemporaryDirectory(
            prefix=f".{out_dir.name}-", dir=out_dir.parent
        ) as temporary:
            staged = Path(temporary) / "run"
            snapshot = Path(temporary) / "config_snapshot.toml"
            snapshot.write_bytes(config_bytes)
            logger.info("Running experiment from %s...", config_path)
            started = time.perf_counter()
            comparison = Composer.from_toml(
                snapshot,
                log_interval=getattr(args, "log_interval", None),
                configure_log=not getattr(args, "_cli_logging_configured", False),
                write_artifacts=False,
                artifact_root=staged / "exports",
            )
            _dump_results(comparison, staged, snapshot)
            if out_dir.exists() or out_dir.is_symlink():
                raise ValueError(f"Result directory appeared during execution: {out_dir}")
            staged.rename(out_dir)
            logger.info(
                "Experiment complete in %.2fs. Results saved to %s",
                time.perf_counter() - started,
                out_dir,
            )
    finally:
        lock.rmdir()
    return 0


def handle_run(args: argparse.Namespace) -> int:
    """Handle `mjax run`."""
    return _run_comparison(args)


def handle_parity(args: argparse.Namespace) -> int:
    """Handle `mjax parity`."""
    return _run_comparison(args)


from malthusjax.benchmarking.results import (
    ComparisonResult,
    ExperimentResult,
    MetaComparison,
    RunResult,
)


def _load_comparison(results_dir: Path) -> ComparisonResult:
    """Rebuild a ComparisonResult from the disk schema."""
    data_dir = results_dir / "data"
    pipelines = {}
    negate_map = {}
    manifest_path = results_dir / "metadata" / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported run manifest version in {manifest_path}")
        expected = set()
        for name, entry in manifest["pipelines"].items():
            _validate_pipeline_name(name)
            negate_map[name] = entry["maximize"]
            runs = []
            for seed, relative in entry["records"].items():
                if relative != f"data/pipeline_{name}/seed_{int(seed)}.json":
                    raise ValueError(f"Invalid seed record path in run manifest: {relative}")
                expected.add(relative)
                path = results_dir / relative
                if not path.is_file():
                    raise ValueError(f"Missing manifest seed record: {relative}")
                run = RunResult.from_dict(json.loads(path.read_text()))
                if run.seed != int(seed):
                    raise ValueError(f"Seed does not match manifest record: {relative}")
                runs.append(run)
            pipelines[name] = ExperimentResult(name=name, runs=runs)
        actual = {
            path.relative_to(results_dir).as_posix() for path in data_dir.rglob("seed_*.json")
        }
        if actual != expected:
            raise ValueError(
                f"Unexpected seed records outside run manifest: {sorted(actual - expected)}"
            )
    else:
        logger.warning(
            "Legacy result directory %s has no run manifest. Seed membership is unverified.",
            results_dir,
        )
        for pipe_dir in sorted(data_dir.glob("pipeline_*")):
            pipe_name = pipe_dir.name.removeprefix("pipeline_")
            runs = [
                RunResult.from_dict(json.loads(path.read_text()))
                for path in sorted(pipe_dir.glob("seed_*.json"))
            ]
            pipelines[pipe_name] = ExperimentResult(name=pipe_name, runs=runs)
    return ComparisonResult(
        pipelines=pipelines, shared_config={}, initial_population=None, negate_map=negate_map
    )


def handle_analyze(args: argparse.Namespace) -> int:
    """Handle `mjax analyze`."""
    results_dir = args.results_dir
    logger.info("Analyzing results in %s...", results_dir)
    comparison = _load_comparison(results_dir)

    pipe_names = list(comparison.pipelines.keys())

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    if len(pipe_names) == 2:
        try:
            from malthusjax.stats import (
                ExpectedDirection,
                HypothesisKind,
                MultipleTestingPolicy,
                Sidedness,
                StatisticalComparator,
                StatisticalComparisonSpec,
                paired_dataset_from_comparison,
            )
        except ImportError as err:
            logger.error(
                "Statistical analysis requires optional dependencies (%s). "
                "Please install via: pip install 'malthusjax[stats]'",
                err,
            )
            return 1

        left, right = pipe_names[0], pipe_names[1]
        logger.info("Running paired difference analysis for %s vs %s...", left, right)
        spec = StatisticalComparisonSpec(
            metric_name="best_fitness",
            hypothesis_kind=HypothesisKind("location_shift"),
            sidedness=Sidedness("two_sided"),
            expected_direction=ExpectedDirection("left_lt_right"),
            alpha=0.05,
            multiple_testing=MultipleTestingPolicy("holm"),
        )
        dataset = paired_dataset_from_comparison(comparison, left, right, spec)
        comparator = StatisticalComparator()
        suite = comparator.compare_suite([dataset], spec)

        with open(analysis_dir / "parity_summary.json", "w") as f:
            json.dump(suite.to_dict(), f, indent=2)

        md_text = suite.to_markdown()
        with open(analysis_dir / "parity_summary.md", "w") as f:
            f.write(md_text)

        print("\n--- Paired Difference Summary ---")
        print(md_text)
        print("----------------------\n")
        logger.info("Analysis generated in %s", analysis_dir)
    else:
        # Just standard mean/std dumps
        logger.info("Saving standard mean/std dumps and unified tables...")
        for name, exp in comparison.pipelines.items():
            summary = exp.aggregated_summary()
            with open(analysis_dir / f"{name}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        try:
            import pandas as pd

            table_data: Any = comparison.summary_table()
            if isinstance(table_data, dict):
                formatted_dict: dict[str, dict[str, str]] = {}
                for pipe, metrics in table_data.items():
                    if isinstance(metrics, dict):
                        formatted_dict[pipe] = {}
                        for k, v in metrics.items():
                            if isinstance(v, dict) and v.get("ci_margin", 0.0) > 0.0:
                                formatted_dict[pipe][k] = f"{v['mean']:.4g} ± {v['ci_margin']:.4g}"
                            elif isinstance(v, dict):
                                formatted_dict[pipe][k] = f"{v['mean']:.4g}"
                df = pd.DataFrame(formatted_dict).T
                df.to_csv(analysis_dir / "comparison_table.csv")
                md_text = str(df.to_markdown())
                with open(analysis_dir / "comparison_table.md", "w") as f:
                    f.write(md_text)

                print("\n--- Comparison Summary ---")
                print(md_text)
                print("--------------------------\n")

            latex_str = comparison.summary_table(latex=True)
            if isinstance(latex_str, str):
                with open(analysis_dir / "comparison_table.tex", "w") as f:
                    f.write(latex_str)
        except Exception as e:
            logger.warning("Could not generate unified tables: %s", e)

    return 0


def handle_plot(args: argparse.Namespace) -> int:
    """Handle `mjax plot`."""
    results_dir = args.results_dir
    logger.info("Plotting results for %s...", results_dir)

    comparison = _load_comparison(results_dir)
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Try to generate the combined convergence plot
    try:
        comparison.plot_convergence(save_path=plot_dir / "convergence.png")
        logger.info("Generated %s", plot_dir / "convergence.png")
    except Exception as e:
        logger.warning("Could not generate convergence plot: %s", e)

    # Try to generate the boxplot comparison
    try:
        comparison.plot_boxplots(save_path=plot_dir / "fitness_distribution.png")
        logger.info("Generated %s", plot_dir / "fitness_distribution.png")
    except Exception as e:
        logger.warning("Could not generate boxplots: %s", e)

    # Try to generate the timings boxplot
    try:
        comparison.plot_boxplots(metric_key="execution", save_path=plot_dir / "timings.png")
        logger.info("Generated %s", plot_dir / "timings.png")
    except Exception as e:
        logger.warning("Could not generate timings boxplot: %s", e)

    return 0


def handle_report(args: argparse.Namespace) -> int:
    """Handle `mjax report`."""
    logger.info("Generating full report for %s", args.results_dir)
    res = handle_analyze(args)
    if res != 0:
        return res
    return handle_plot(args)


def handle_aggregate(args: argparse.Namespace) -> int:
    """Handle `mjax aggregate`."""
    out_dir = args.out_dir
    results_dirs = args.results_dirs

    logger.info("Aggregating %d experiments into %s...", len(results_dirs), out_dir)

    comparisons = {}
    for d in results_dirs:
        logger.debug("Loading %s...", d)
        try:
            comp = _load_comparison(d)
            if comp.pipelines:
                comparisons[d.name] = comp
            else:
                logger.warning("No pipelines found in %s", d)
        except Exception as e:
            logger.warning("Failed to load %s: %s", d, e)

    if not comparisons:
        logger.error("No valid experiments loaded.")
        return 1

    meta = MetaComparison(comparisons)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    logger.info("Generating aggregate convergence grid...")
    meta.plot_convergence_grid(save_path=plot_dir / "convergence_grid.png")

    logger.info("Generating aggregate boxplot grid...")
    meta.plot_boxplot_grid(save_path=plot_dir / "fitness_distribution_grid.png")

    logger.info("Generating aggregate timings grid...")
    try:
        meta.plot_boxplot_grid(metric_key="execution", save_path=plot_dir / "timings_grid.png")
    except ValueError as exc:
        logger.warning("Timing grid unavailable: %s", exc)

    logger.info("Generating aggregate summary JSON...")
    summary = meta.summary_table()
    with open(out_dir / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Aggregate report complete. Results saved in %s", out_dir)
    return 0


def handle_catalog(args: argparse.Namespace) -> int:
    """Handle `mjax catalog`."""
    catalog = OperatorCatalog()
    available = catalog.list_available()
    print("--- MalthusJAX Operator Catalog ---")
    for key in sorted(available):
        print(f"  - {key}")
    return 0


def _show_skill(args: argparse.Namespace) -> int:
    from malthusjax.skill import main as skill_main

    return skill_main([])


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    log_parser = argparse.ArgumentParser(add_help=False)
    log_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable verbose debug logging (DEBUG level)",
    )
    log_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Suppress informational logging (WARNING level)",
    )
    log_parser.add_argument(
        "--log-file",
        type=Path,
        default=argparse.SUPPRESS,
        help="Destination file path for structured logs",
    )
    log_parser.add_argument(
        "--log-json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Format console and file logs as JSON lines",
    )
    log_parser.add_argument(
        "--log-interval",
        type=int,
        default=argparse.SUPPRESS,
        help="Interval in generations for JIT telemetry callbacks",
    )
    log_parser.add_argument(
        "--no-crash-handler",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable native POSIX crash handler trap",
    )

    parser = argparse.ArgumentParser(
        prog="mjax",
        description="MalthusJAX Unified Benchmarking & Analysis CLI",
        parents=[log_parser],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. RUN
    parser_run = subparsers.add_parser(
        "run", parents=[log_parser], help="Run an experiment from a TOML config"
    )
    parser_run.add_argument("config", type=Path, help="Path to experiment TOML config")
    parser_run.add_argument("--output-dir", type=Path, help="Fresh result directory")
    parser_run.set_defaults(func=handle_run)

    # 2. PARITY
    parser_parity = subparsers.add_parser(
        "parity",
        parents=[log_parser],
        help="Run seed-aligned statistical parity between pipelines",
    )
    parser_parity.add_argument("config", type=Path, help="Path to parity TOML config")
    parser_parity.add_argument("--output-dir", type=Path, help="Fresh result directory")
    parser_parity.set_defaults(func=handle_parity)

    # 3. ANALYZE
    parser_analyze = subparsers.add_parser(
        "analyze",
        parents=[log_parser],
        help="Calculate statistical summaries from raw data",
    )
    parser_analyze.add_argument("results_dir", type=Path, help="Directory containing raw JSON data")
    parser_analyze.set_defaults(func=handle_analyze)

    # 4. PLOT
    parser_plot = subparsers.add_parser(
        "plot",
        parents=[log_parser],
        help="Generate diagnostic plots from raw data and analysis",
    )
    parser_plot.add_argument("results_dir", type=Path, help="Directory containing raw JSON data")
    parser_plot.set_defaults(func=handle_plot)

    # 5. REPORT (Analyze + Plot)
    parser_report = subparsers.add_parser(
        "report",
        parents=[log_parser],
        help="Generate both statistical summaries and diagnostic plots",
    )
    parser_report.add_argument("results_dir", type=Path, help="Directory containing raw JSON data")
    parser_report.set_defaults(func=handle_report)

    # 6. AGGREGATE
    parser_aggregate = subparsers.add_parser(
        "aggregate",
        parents=[log_parser],
        help="Aggregate multiple experiments into a suite report",
    )
    parser_aggregate.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for the aggregate suite",
    )
    parser_aggregate.add_argument(
        "results_dirs",
        type=Path,
        nargs="+",
        help="One or more experiment result directories",
    )
    parser_aggregate.set_defaults(func=handle_aggregate)

    skill_parser = subparsers.add_parser("skill", help="Print the bundled agent skill")
    skill_parser.set_defaults(func=_show_skill)

    # 7. CATALOG
    parser_catalog = subparsers.add_parser(
        "catalog",
        parents=[log_parser],
        help="List registered framework operators",
    )
    parser_catalog.set_defaults(func=handle_catalog)

    # 8. CHECK-ENV
    parser_check_env = subparsers.add_parser(
        "check-env",
        parents=[log_parser],
        help="Inspect runtime environment, HPC thread limits, and JAX GPU configuration",
    )
    parser_check_env.set_defaults(func=handle_check_env)

    # 9. Discovered extension commands (e.g., gp, neat, or third-party extensions)
    from malthusjax.composer.plugins import discover_cli_commands

    for cmd_name, cmd_entrypoint in discover_cli_commands().items():

        def _make_handler(ep: Any) -> Callable[[argparse.Namespace], int]:
            return lambda parsed_args: int(ep(getattr(parsed_args, "sub_args", [])))

        ext_parser = subparsers.add_parser(
            cmd_name,
            help=f"{cmd_name.upper()} extension commands",
        )
        ext_parser.add_argument(
            "sub_args",
            nargs=argparse.REMAINDER,
            help=f"Arguments forwarded to {cmd_name}",
        )
        ext_parser.set_defaults(func=_make_handler(cmd_entrypoint))

    parsed = parser.parse_args(args)

    verbose = getattr(parsed, "verbose", False)
    quiet = getattr(parsed, "quiet", False)
    log_file = getattr(parsed, "log_file", None)
    log_json = getattr(parsed, "log_json", False)
    log_interval = getattr(parsed, "log_interval", None)
    no_crash_handler = getattr(parsed, "no_crash_handler", False)

    setattr(parsed, "verbose", verbose)
    setattr(parsed, "quiet", quiet)
    setattr(parsed, "log_file", log_file)
    setattr(parsed, "log_json", log_json)
    setattr(parsed, "log_interval", log_interval)
    setattr(parsed, "no_crash_handler", no_crash_handler)

    if no_crash_handler:
        uninstall_crash_handler()

    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = "INFO"

    format_type = "json" if log_json else "color"
    configure_logging(level=level, log_file=log_file, format_type=format_type)
    setattr(parsed, "_cli_logging_configured", True)

    try:
        return parsed.func(parsed)
    except Exception as e:
        logger.error("Command failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
