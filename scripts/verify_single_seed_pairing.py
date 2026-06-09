#!/usr/bin/env python3
"""Verify single-seed behavior: toy_gap_convergence vs Composer.compare

Compares final gap for both backends on one seed and prints results.
"""
from __future__ import annotations

import subprocess
import json
from pathlib import Path

import jax.random as jr

from malthusjax.composer import Composer


def run_toy(seed: int):
    cmd = [
        "python", "examples/toy_gap_convergence.py", "--backend", "both",
        "--function", "sphere", "--dimensions", "5", "--pop-size", "12",
        "--generations", "20", "--seed", str(seed),
        "--elite-k", "2", "--crossover-rate", "0.3", "--mutation-strength", "0.05"
    ]
    print("Running toy_gap_convergence.py (both backends)...")
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout)
    return out.stdout


def run_composer_compare(seed: int):
    composer = Composer.create_default()
    pipelines = {
        "MJX": {
            "backend": "malthusjax",
            "selection": "elite_pool:num_selections=12,elite_k=2",
            "crossover": "evosax_uniform_crossover:crossover_rate=0.3",
            "mutation": "evosax_gaussian:mutation_strength=0.05",
            "elitism": 0,
        },
        "EvoSAX": {
            "backend": "evosax",
            "evosax_strategy": "SimpleGA",
            "strategy_params": {"crossover_rate": 0.3, "elite_ratio": 2/12.0, "mutation_std": 0.05},
        },
    }

    print("Running Composer.compare(...) with shared_initial_population=True")
    comp = composer.compare(pipelines=pipelines, seeds=(seed,), shared_initial_population=True, pop_seed=seed, fitness=f"bbob:fn_name=sphere,num_dims=5,maximize=false", pop_size=12, generations=20)

    results = {}
    for name, exp in comp.pipelines.items():
        run = exp.runs[0]
        # last history entry
        last = run.history[-1] if run.history else {}
        best = last.get('best_fitness', run.metrics.get('best_fitness'))
        results[name] = best
    return results


if __name__ == '__main__':
    seed = 0
    toy_out = run_toy(seed)
    comp_res = run_composer_compare(seed)
    print('\nComposer.compare results:')
    print(json.dumps(comp_res, indent=2))
