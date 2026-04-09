import time

from malthusjax.composer import Composer
from malthusjax.composer.config import load_experiment_config


def main() -> None:
    print("Testing Composer with TOML configuration...")

    meta, pipelines = load_experiment_config("examples/mock_binary_experiment.toml")
    composer = Composer()

    print(f"Loaded config for {meta.get('name')}")

    for pipeline_name, kwargs in pipelines.items():
        print(f"\nRunning pipeline: {pipeline_name}")
        start_t = time.time()

        # kwargs already has seeds, generations, operators, experiment_name from TOML
        result = composer.quick_run(
            output_dir=meta.get("output_dir"),
            **kwargs
        )
        end_t = time.time()

        run_metrics = result.runs[0].metrics
        best_fit = run_metrics.get("best_fitness", "N/A")

        print("\n--- PIPELINE DONE ---")
        if best_fit != "N/A" and hasattr(best_fit, "item"):
            print(f"Global Best Fitness: {best_fit.item():.4f}")
        else:
            print(f"Global Best Fitness: {best_fit}")
        print(f"Execution Time    : {end_t - start_t:.4f}s")
        if result.runs[0].error:
            print(f"ERROR: {result.runs[0].error}")

if __name__ == "__main__":
    main()
