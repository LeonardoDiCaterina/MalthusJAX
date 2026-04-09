"""Master thesis benchmarking runner.

Discovers all convergence_*.toml and scaling_*.toml files in the examples directory,
runs each experiment with Composer.from_toml(), generates high-resolution plots,
and exports summary tables for thesis writing.

Usage:
    python run_all_thesis_experiments.py              # Run all experiments
    python run_all_thesis_experiments.py sphere_dim10 # Run specific pattern
    nohup python run_all_thesis_experiments.py > thesis_bench.log 2>&1 &
"""

from pathlib import Path
import sys
import json
from datetime import datetime
import argparse

from malthusjax.composer import Composer
import matplotlib.pyplot as plt


def discover_toml_files(pattern="convergence_"):
    """Find all TOML files matching pattern in examples directory."""
    examples_dir = Path(__file__).resolve().parent
    tomls = sorted(examples_dir.glob(f"{pattern}*.toml"))
    return tomls


def run_single_experiment(toml_path, skip_plots=False):
    """Run a single TOML experiment and generate results."""
    print(f"\n{'='*70}")
    print(f"Running: {toml_path.name}")
    print(f"{'='*70}")
    start_time = datetime.now()

    # Load and run with Composer
    try:
        comparison = Composer.from_toml(
            str(toml_path),
            shared_initial_population=True,
            pop_seed=123,
        )
    except Exception as e:
        print(f"ERROR loading TOML: {e}")
        return None

    print(f"Pipelines loaded: {comparison.names}")

    # Get experiment name from TOML filename
    exp_name = toml_path.stem  # e.g., "convergence_sphere_dim10"

    # Create result directory
    result_dir = Path(__file__).resolve().parent / "results" / "thesis" / exp_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    summary = comparison.summary_table()
    print("\nAggregated Summary:")
    for pipeline_name, metrics in summary.items():
        best_fitness = metrics.get("best_fitness", "N/A")
        print(f"  {pipeline_name:30s} best_fitness={best_fitness:10.6f}")

    if skip_plots:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Completed (plots skipped) in {elapsed:.1f}s")
        return result_dir

    # Generate plots
    print("\nGenerating plots...")
    try:
        # Convergence plot - multiple seeds
        seed_list = [0, 1, 2, 3]
        axes = comparison.plot_convergence(
            seed_index=seed_list,
            save_path=result_dir / "convergence_seeds_0-3.png",
        )
        for ax in axes:
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best Fitness")
        plt.suptitle(f"Convergence: {exp_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(result_dir / "convergence_seeds_0-3.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ convergence_seeds_0-3.png")
    except Exception as e:
        print(f"  ✗ convergence plot failed: {e}")

    try:
        # Timing boxplot
        ax_timing = comparison.plot_timing_boxplot(
            timing_key="duration_seconds",
            save_path=result_dir / "timing_boxplot.png",
        )
        ax_timing.set_title("Per-pipeline Duration Distribution")
        ax_timing.set_xticklabels(ax_timing.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(result_dir / "timing_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ timing_boxplot.png")
    except Exception as e:
        print(f"  ✗ timing boxplot failed: {e}")

    try:
        # Final metric boxplot
        ax_final = comparison.plot_final_metric_boxplot(
            metric_key="best_fitness",
            save_path=result_dir / "final_best_fitness_boxplot.png",
        )
        ax_final.set_title("Final Best Fitness Distribution")
        ax_final.set_xticklabels(ax_final.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(result_dir / "final_best_fitness_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ final_best_fitness_boxplot.png")
    except Exception as e:
        print(f"  ✗ final metric boxplot failed: {e}")

    # Export summary table
    try:
        latex_table = comparison.summary_table(latex=True)
        with open(result_dir / "summary_table.tex", "w") as f:
            f.write(latex_table)
        print("  ✓ summary_table.tex")
    except Exception as e:
        print(f"  ✗ LaTeX table failed: {e}")

    # Export convergence data
    try:
        conv_data = comparison.convergence_data(seed_index=0)
        import json
        conv_export = {}
        for pipeline_name, history in conv_data.items():
            conv_export[pipeline_name] = history
        with open(result_dir / "convergence_seed_0.json", "w") as f:
            json.dump(conv_export, f, indent=2)
        print("  ✓ convergence_seed_0.json")
    except Exception as e:
        print(f"  ✗ convergence export failed: {e}")

    # Export summary stats
    try:
        summary_stats = {}
        for pipeline_name in comparison.names:
            agg = comparison.pipelines[pipeline_name].aggregated_summary()
            summary_stats[pipeline_name] = agg
        with open(result_dir / "aggregated_summary.json", "w") as f:
            json.dump(summary_stats, f, indent=2, default=str)
        print("  ✓ aggregated_summary.json")
    except Exception as e:
        print(f"  ✗ summary export failed: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed:.1f}s → {result_dir}")
    return result_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run all thesis benchmarking experiments"
    )
    parser.add_argument(
        "--pattern",
        default="convergence_",
        help="TOML filename pattern (default: convergence_)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation (faster for testing)",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Run only TOML file matching this name (e.g., sphere_dim10)",
    )
    args = parser.parse_args()

    # Discover TOML files
    tomls = discover_toml_files(pattern=args.pattern)

    if not tomls:
        print(f"No TOML files found matching pattern: {args.pattern}")
        return 1

    if args.single:
        tomls = [t for t in tomls if args.single in t.name]
        if not tomls:
            print(f"No TOML files found matching: {args.single}")
            return 1

    print(f"Discovered {len(tomls)} experiment(s)")
    for toml in tomls:
        print(f"  - {toml.name}")

    # Run each
    results = []
    for toml_path in tomls:
        try:
            result_dir = run_single_experiment(toml_path, skip_plots=args.skip_plots)
            if result_dir:
                results.append(result_dir)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 130
        except Exception as e:
            print(f"FATAL error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary report
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print("Results saved to:")
    for result_dir in results:
        print(f"  {result_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
