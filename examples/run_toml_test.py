import argparse
import sys
import time

from malthusjax.composer import Composer
from malthusjax.composer.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MalthusJAX experiments from TOML configuration files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/mock_binary_experiment.toml",
        help="Path to TOML configuration file (default: examples/mock_binary_experiment.toml)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help="Run only a specific pipeline (optional; runs all if not specified)",
    )
    args = parser.parse_args()

    print(f"Loading TOML configuration: {args.config}")
    try:
        # load_experiment_config returns ExperimentLoadResult with: meta, pipelines, data_registry
        result = load_experiment_config(
            args.config, pipelines=[args.pipeline] if args.pipeline else None
        )
        meta = result.meta
        pipelines = result.pipelines
        data_registry = result.data_registry
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    print(f"✓ Loaded config: {meta.get('name')}")
    print(f"  Output dir: {meta.get('output_dir')}")
    print(f"  Running {len(pipelines)} pipeline(s)...\n")

    composer = Composer()

    total_time = 0.0
    results_summary = []

    for pipeline_name, kwargs in pipelines.items():
        print(f"→ Pipeline: {pipeline_name}")
        start_t = time.time()

        # kwargs already has seeds, generations, operators, experiment_name from TOML
        try:
            result = composer.quick_run(
                output_dir=meta.get("output_dir"),
                data_config=data_registry,
                **kwargs
            )
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:100]}")
            results_summary.append((pipeline_name, "ERROR", 0.0, str(e)[:50]))
            continue

        end_t = time.time()
        elapsed = end_t - start_t
        total_time += elapsed

        run_metrics = result.runs[0].metrics
        best_fit = run_metrics.get("best_fitness", "N/A")

        if best_fit != "N/A" and hasattr(best_fit, "item"):
            best_fit_val = float(best_fit.item())
        else:
            best_fit_val = best_fit

        status = "✓" if not result.runs[0].error else "✗"
        results_summary.append((pipeline_name, status, elapsed, best_fit_val))

        print(f"  {status} Best: {best_fit_val:.6f} | Time: {elapsed:.2f}s")
        if result.runs[0].error:
            print(f"  Error: {result.runs[0].error}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status, elapsed, fitness in results_summary:
        fit_str = f"{fitness:.6f}" if isinstance(fitness, float) else str(fitness)
        print(f"{status} {name:30s} | Fitness: {fit_str:15s} | Time: {elapsed:7.2f}s")
    print(f"\nTotal time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
