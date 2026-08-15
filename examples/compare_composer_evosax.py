#!/usr/bin/env python3
"""Compare MalthusJAX and Evosax backends using Composer and plot timing results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from malthusjax.composer import Composer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MalthusJAX and Evosax backends with Composer."
    )
    parser.add_argument("--fitness", type=str, default="sphere:dim=10")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, default=Path("results/composer_compare"))
    parser.add_argument("--plot-convergence", action="store_true", help="Save convergence plot")
    parser.add_argument("--plot-timings", action="store_true", help="Save timing bar charts")
    return parser.parse_args()


def build_pipelines() -> dict[str, dict[str, Any]]:
    return {
        "MalthusJAX:Blend+Gaussian": {
            "backend": "malthusjax",
            "selection": "tournament:num_selections=25,tournament_size=3",
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.2,mutation_strength=0.1",
        },
        "Evosax:SimpleGA": {
            "backend": "evosax",
            "evosax_strategy": "SimpleGA",
        },
    }


def timings_dataframe(comparison) -> pd.DataFrame:
    rows = []
    for name, exp in comparison.pipelines.items():
        for run in exp.runs:
            row = {
                "pipeline": name,
                "seed": run.seed,
                "duration_seconds": run.duration_seconds,
            }
            if run.timings:
                row.update(run.timings)
            rows.append(row)
    return pd.DataFrame(rows)


def save_timing_plots(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    numeric_cols = [c for c in df.columns if c not in {"pipeline", "seed"}]
    mean_df = df.groupby("pipeline")[numeric_cols].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    mean_df.plot.bar(x="pipeline", y="duration_seconds", ax=ax, legend=False)
    ax.set_ylabel("Mean Duration (s)")
    ax.set_title("Average Experiment Duration by Pipeline")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "timings_duration.png")
    plt.close(fig)

    timing_cols = [c for c in numeric_cols if c != "duration_seconds"]
    if timing_cols:
        mean_df = mean_df[["pipeline"] + timing_cols]
        mean_df = mean_df.set_index("pipeline")
        fig, ax = plt.subplots(figsize=(10, 5))
        mean_df.plot.bar(ax=ax)
        ax.set_ylabel("Mean Time (s)")
        ax.set_title("Mean Timing Breakdown by Pipeline")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "timings_breakdown.png")
        plt.close(fig)


def save_convergence_plot(comparison, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison.plot_convergence(seed_index=0, ax=ax, title="Seed 0 Convergence")
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_seed_0.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    composer = Composer.create_default()
    pipelines = build_pipelines()

    comparison = composer.compare(
        pipelines=pipelines,
        fitness=args.fitness,
        pop_size=args.pop_size,
        generations=args.generations,
        seeds=tuple(args.seeds),
        shared_initial_population=True,
    )

    summary = comparison.summary_table()
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(output_dir / "comparison_summary.csv")
    print("Saved summary table to:", output_dir / "comparison_summary.csv")
    print(summary_df)

    df = timings_dataframe(comparison)
    df.to_csv(output_dir / "timings.csv", index=False)
    print("Saved raw timing data to:", output_dir / "timings.csv")

    if args.plot_timings:
        save_timing_plots(df, output_dir)
        print("Saved timing plot(s).")

    if args.plot_convergence:
        save_convergence_plot(comparison, output_dir)
        print("Saved convergence plot(s).")

    print("Done.")


if __name__ == "__main__":
    main()
