#!/usr/bin/env python3
"""Hyperparameter sweep for Evosax SimpleGA using MalthusJAX evosax adapter.

This script explores SimpleGA's `crossover_rate`, `std_schedule`, and
`elite_ratio` parameters to determine whether any combination can reach
performance parity with the MalthusJAX evosax-operator baseline.

Run from the repository root:
    python scripts/simplega_grid_search.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.random as jr
import numpy as np
import optax

from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SimpleGA hyperparameters.")
    parser.add_argument("--output", type=Path, default=None, help="Write top results to JSON file.")
    parser.add_argument("--pop-size", type=int, default=1024)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--dims", type=int, default=10)
    parser.add_argument("--function", type=str, default="rastrigin")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator = BBOBEvaluator.create(
        BBOBConfig(fn_name=args.function, num_dims=args.dims, seed=0, maximize=False)
    )

    stds = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    crs = [0.0, 0.2, 0.5, 0.8]
    elite_ratios = [0.5, 0.2, 0.1, 0.05, 0.02]

    results = []
    for std in stds:
        for cr in crs:
            for er in elite_ratios:
                values = []
                for seed in args.seeds:
                    engine = build_evosax_engine(
                        strategy_name="SimpleGA",
                        evaluator=evaluator,
                        pop_size=args.pop_size,
                        generations=args.generations,
                        bounds=(-5.0, 5.0),
                        maximize=False,
                        seed=seed,
                        strategy_params={
                            "crossover_rate": cr,
                            "elite_ratio": er,
                            "std_schedule": optax.constant_schedule(std),
                        },
                    )
                    result = engine.run_once(jr.PRNGKey(seed), compile=False)
                    values.append(float(result["summary"]["best_fitness"]))

                mean = float(np.mean(values))
                stddev = float(np.std(values, ddof=1))
                results.append(
                    {
                        "std": std,
                        "crossover_rate": cr,
                        "elite_ratio": er,
                        "mean_best_fitness": mean,
                        "stddev_best_fitness": stddev,
                        "seed_values": values,
                    }
                )

    results.sort(key=lambda entry: entry["mean_best_fitness"], reverse=True)
    top10 = results[:10]

    print("Top 10 SimpleGA hyperparameter combos (higher is better):")
    for row in top10:
        print(
            f"std={row['std']:.3f} cr={row['crossover_rate']:.1f} er={row['elite_ratio']:.2f} "
            f"mean={row['mean_best_fitness']:.4f} std={row['stddev_best_fitness']:.4f} "
            f"values={row['seed_values']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Wrote full results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
