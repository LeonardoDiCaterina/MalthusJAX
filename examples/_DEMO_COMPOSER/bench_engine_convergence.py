"""Composer Demo: Convergence Comparison with Canonical Initialization

This script compares convergence across multiple MalthusJAX pipelines using a
shared initial population and a fixed set of seeds. The configuration is defined
in `bench_engine_convergence.toml` and loaded with `Composer.from_toml()`.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from malthusjax.composer import Composer


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    toml_path = project_dir / "bench_engine_convergence.toml"

    print(f"Loading TOML from: {toml_path}")
    print("Using canonical shared initialization and Composer.from_toml()")

    seeds = tuple(range(42, 142))  # 100 seeds
    print(f"Configured {len(seeds)} seeds")

    comparison = Composer.from_toml(
        str(toml_path),
        shared_initial_population=True,
        pop_seed=123,
    )

    print(f"Pipelines loaded: {comparison.names}")
    print(f"Seeds used: {len(seeds)}")

    summary = comparison.summary_table()
    print("Aggregated summary:")
    for pipeline_name, metrics in summary.items():
        best_fitness = metrics.get("best_fitness")
        print(f"- {pipeline_name}: best_fitness mean={best_fitness:.6f}")

    seed_index = 0
    seed_history = comparison.convergence_data(seed_index=seed_index)
    print(f"\nConvergence history available for seed {seed_index}:")
    for pipeline_name, history in seed_history.items():
        print(f"- {pipeline_name}: {len(history)} generations")

    seed_count = min(4, len(seeds))
    seed_list = list(range(seed_count))

    axes = comparison.plot_convergence(seed_index=seed_list)
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")

    plt.suptitle("Convergence comparison across multiple shared-seed runs", fontsize=16)
    plt.show()

    ax_box = comparison.plot_timing_boxplot(timing_key="duration_seconds")
    ax_box.set_title("Per-pipeline duration boxplot")
    ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    ax_final = comparison.plot_final_metric_boxplot(metric_key="best_fitness")
    ax_final.set_title("Final best_fitness distribution across runs")
    ax_final.set_xticklabels(ax_final.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
