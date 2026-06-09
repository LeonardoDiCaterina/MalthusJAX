#!/usr/bin/env python3
"""Compare Evosax vs MalthusJAX across different offspring counts.

Creates one Evosax pipeline and N MalthusJAX pipelines that only differ
by the `num_offspring` crossover parameter and runs `Composer.compare`.

Usage:
  python scripts/run_offspring_sweep.py --pop-size 64 --generations 50 --offspring 1 2 3

Output: writes `results/offspring_sweep_report.json` with aggregated summaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from malthusjax.composer import Composer


def objective_optimum_value(fitness_spec: str) -> float | None:
    spec = fitness_spec.split(":", 1)[0].strip().lower()
    if spec in {"sphere", "rastrigin", "griewank", "bbob"}:
        return 0.0
    return None


def paired_permutation_test(
    a: list[float],
    b: list[float],
    n_perm: int = 10000,
    alternative: str = "two-sided",
) -> float:
    import random

    if len(a) != len(b):
        raise ValueError("Samples must be same length for paired test")
    diffs = [x - y for x, y in zip(a, b)]
    obs = abs(sum(diffs))
    count = 0
    for _ in range(n_perm):
        signs = [random.choice([1, -1]) for _ in diffs]
        perm = sum(s * d for s, d in zip(signs, diffs))
        if alternative == "two-sided":
            hit = abs(perm) >= obs
        elif alternative == "less":
            hit = perm <= -obs
        elif alternative == "greater":
            hit = perm >= obs
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        if hit:
            count += 1
    return (count + 1) / (n_perm + 1)


def build_pipelines(base_cfg: Dict[str, Any], offspring_counts: list[int], evosax_strategy: str):
    pipelines: Dict[str, Dict[str, Any]] = {}
    # MalthusJAX pipelines differing only by num_offspring in crossover spec
    for n in offspring_counts:
        name = f"malthus_num_offspring_{n}"
        # keep other operator specs minimal and consistent
        crossover_spec = f"uniform_real:crossover_rate=0.5,num_offspring={n}"
        mutation_spec = "gaussian:mutation_rate=0.1,mutation_strength=0.05,num_offspring=1"
        pipelines[name] = dict(
            backend="malthusjax",
            crossover=crossover_spec,
            mutation=mutation_spec,
        )

    # Evosax parity variants: match crossover/mutation rates and offspring semantics
    for n in offspring_counts:
        name = f"evosax_{evosax_strategy}_num_offspring_{n}"
        strategy_params = {
            "crossover_rate": 0.5,
            # alias for mutation strength (handled by builder)
            "mutation_std": 0.05,
        }
        pipelines[name] = dict(
            backend="evosax",
            evosax_strategy=evosax_strategy,
            strategy_params=strategy_params,
        )

    return pipelines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pop-size", type=int, default=64)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--fitness", type=str, default="sphere:dim=5")
    p.add_argument("--seeds", type=int, nargs="*", default=(1, 2, 3))
    p.add_argument("--offspring", type=int, nargs="*", default=(1, 2, 3))
    p.add_argument("--evosax-strategy", type=str, default="SimpleGA")
    p.add_argument("--output-dir", type=str, default="results")
    args = p.parse_args()

    composer = Composer.create_default()

    shared_kwargs = dict(
        fitness=args.fitness,
        pop_size=args.pop_size,
        generations=args.generations,
    )

    pipelines = build_pipelines(shared_kwargs, list(args.offspring), args.evosax_strategy)

    print(f"Running comparison for pipelines: {list(pipelines.keys())}")
    comp = composer.compare(
        pipelines=pipelines,
        seeds=tuple(args.seeds),
        shared_initial_population=True,
        pop_seed=123,
        **shared_kwargs,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    report = {"pipelines": {}}
    for name, res in comp.pipelines.items():
        report["pipelines"][name] = res.aggregated_summary()

    outpath = outdir / "offspring_sweep_report.json"
    with outpath.open("w") as f:
        json.dump(report, f, indent=2)

    # Save aggregated report
    print(f"Wrote report to {outpath}")

    # Save detailed per-seed final metric values
    detailed = {"final_values": comp.final_metric_data(metric_key="best_fitness")}
    detpath = outdir / "offspring_sweep_detailed.json"
    with detpath.open("w") as f:
        json.dump(detailed, f, indent=2)

    stats_results = {}
    optimum = objective_optimum_value(args.fitness)
    for n in args.offspring:
        mname = f"malthus_num_offspring_{n}"
        ename = f"evosax_{args.evosax_strategy}_num_offspring_{n}"
        a = detailed["final_values"].get(mname, [])
        b = detailed["final_values"].get(ename, [])
        if not a or not b or len(a) != len(b):
            stats_results[f"n_{n}"] = {"error": "missing or unequal-length samples"}
            continue
        try:
            from scipy import stats

            tstat, pval = stats.ttest_rel(a, b)
            entry = {"raw": {"method": "ttest_rel", "t": tstat, "p": pval}}
        except Exception:
            p = paired_permutation_test(a, b, n_perm=10000)
            entry = {"raw": {"method": "permutation", "p": p}}

        if optimum is not None:
            da = [abs(float(x) - optimum) for x in a]
            db = [abs(float(x) - optimum) for x in b]
            alt = "less" if (sum(da) / len(da)) < (sum(db) / len(db)) else "greater"
            try:
                from scipy import stats

                wstat, wpval = stats.wilcoxon(da, db, alternative=alt)
                entry["distance_to_optimum"] = {"method": "wilcoxon", "w": wstat, "p": wpval, "alternative": alt, "optimum": optimum}
            except Exception:
                p = paired_permutation_test(da, db, n_perm=10000, alternative=alt)
                entry["distance_to_optimum"] = {"method": "permutation", "p": p, "alternative": alt, "optimum": optimum}

        stats_results[f"n_{n}"] = entry

    statspath = outdir / "offspring_sweep_stats.json"
    with statspath.open("w") as f:
        json.dump(stats_results, f, indent=2)

    print(f"Wrote detailed results to {detpath}")
    print(f"Wrote statistics to {statspath}")


if __name__ == "__main__":
    main()
