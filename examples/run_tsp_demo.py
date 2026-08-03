"""Example: Running TSP optimization with Option C (Data IDs)."""

from malthusjax.composer import Composer
from malthusjax.composer.config import load_experiment_config


def main() -> None:
    # 1. Load config which now includes [data.*] sections!
    result = load_experiment_config("examples/tsp_experiment.toml")

    composer = Composer()

    print(f"Running Experiment: {result.meta.get('name')}")
    print(f"Registered Data Sources: {list(result.data_registry.keys())}\\n")

    # 2. Iterate and run each pipeline
    for pipeline_name, kwargs in result.pipelines.items():
        print(f"\\n--- Start Pipeline: {pipeline_name} ---")

        # 3. Pass data_config into quick_run so it can resolve data_ids
        exec_result = composer.quick_run(data_config=result.data_registry, **kwargs)

        print(f"Pipeline {pipeline_name} Complete.")
        print(f"Summary: {exec_result.aggregated_summary()}")


if __name__ == "__main__":
    main()
