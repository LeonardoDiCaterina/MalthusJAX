#!/usr/bin/env python3
"""
Run a comparison between the FAST_LANE (kernel) engine and a LEGACY engine.

This script builds a large GA workload and measures:
- Fast Lane (compile + run) - first (cold) run includes compilation time
- Fast Lane warm run (uses cached compiled function)
- Legacy run (no JIT, Python splitting path)

Usage: adjust defaults via CLI args for quicker local iteration.
"""
import time
import argparse

import jax
import jax.random as jr

from malthusjax import GeneticEngine, GeneticEngineParams, RealGenomeConfig
from malthusjax.core.fitness.real_evaluators import SphereEvaluator, SphereConfig
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.base import BaseSelection
from malthusjax.engine.inspector import ExecutionMode


def build_engine(pop_size, genome_length, elitism, num_generations, legacy_selection_wrapper=False):
    params = GeneticEngineParams(pop_size=pop_size, elitism=elitism, num_generations=num_generations)

    genome_config = RealGenomeConfig(length=genome_length, bounds=(-1.0, 1.0))
    evaluator = SphereEvaluator(SphereConfig(maximize=False))

    mut = GaussianMutation(num_offspring=1, mutation_rate=0.2, mutation_strength=0.05)
    cross = UniformCrossover(num_offspring=1, crossover_rate=0.5)
    sel = ElitePoolSelection(num_selections=pop_size, elite_k=max(4, pop_size // 20))

    if legacy_selection_wrapper:
        # Wrap the selection into a legacy-like object that lacks kernel identity-card methods
        class LegacySelection(BaseSelection):
            def __init__(self, inner):
                self.inner = inner

            def __call__(self, key, fitness):
                return self.inner(key, fitness)

        sel = LegacySelection(sel)

    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
    )

    return engine, params


def run_and_time(engine, params, seed, compile: bool, time_it: bool = True):
    key = jr.PRNGKey(seed)
    state = engine.init_state(key, params)

    t0 = time.perf_counter()
    # request timing from engine as well
    final_state, history, engine_elapsed = engine.run(state, params, compile=compile, time_it=time_it)
    t1 = time.perf_counter()

    wall_time = t1 - t0
    measured = engine_elapsed if engine_elapsed is not None else wall_time
    return measured, wall_time, final_state, history


def main():
    parser = argparse.ArgumentParser(description="Engine FAST_LANE vs LEGACY comparison")
    parser.add_argument("--pop_size", type=int, default=1000)
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Run a quick job with smaller sizes for testing")

    args = parser.parse_args()

    if args.quick:
        pop_size = 128
        genome_length = 64
        generations = 10
    else:
        pop_size = args.pop_size
        genome_length = args.length
        generations = args.generations

    elitism = args.elitism
    seed = args.seed

    print(f"Configuration: pop_size={pop_size}, genome_length={genome_length}, generations={generations}")

    # Build FAST_LANE engine (auto-detect)
    engine_fast, params = build_engine(pop_size, genome_length, elitism, generations)
    print(f"Engine detected mode: {engine_fast.mode}")

    if engine_fast.mode != ExecutionMode.FAST_LANE:
        print("Warning: engine not in FAST_LANE mode; ensure operators are migrated and visible to inspector.")

    # Fast Lane - cold (compile + run)
    print("Running FAST_LANE (cold - includes compilation)")
    fast_cold_measured, fast_cold_wall, _, _ = run_and_time(engine_fast, params, seed, compile=True, time_it=True)
    print(f"FAST_LANE (cold) time: {fast_cold_measured:.4f}s (engine measured), wall: {fast_cold_wall:.4f}s")

    # Fast Lane - warm run (should use cached compiled function)
    print("Running FAST_LANE (warm)")
    fast_warm_measured, fast_warm_wall, _, _ = run_and_time(engine_fast, params, seed + 1, compile=True, time_it=True)
    print(f"FAST_LANE (warm) time: {fast_warm_measured:.4f}s (engine measured), wall: {fast_warm_wall:.4f}s")

    # Build LEGACY engine by wrapping selection to remove kernel identity
    print("Building LEGACY engine by wrapping selection to force fallback")
    engine_legacy, params_legacy = build_engine(pop_size, genome_length, elitism, generations, legacy_selection_wrapper=True)
    print(f"Legacy engine detected mode: {engine_legacy.mode}")

    # Legacy run (no compile)
    print("Running LEGACY (no JIT)")
    legacy_measured, legacy_wall, _, _ = run_and_time(engine_legacy, params_legacy, seed + 2, compile=False, time_it=True)
    print(f"LEGACY time: {legacy_measured:.4f}s (engine measured), wall: {legacy_wall:.4f}s")

    # Compute speedups
    speedup_cold = legacy_measured / fast_cold_measured if fast_cold_measured > 0 else float('inf')
    speedup_warm = legacy_measured / fast_warm_measured if fast_warm_measured > 0 else float('inf')

    print("\n=== Summary ===")
    print(f"Legacy total time (measured): {legacy_measured:.4f}s")
    print(f"Fast Lane cold (compile+run): {fast_cold_measured:.4f}s")
    print(f"Fast Lane warm (cached): {fast_warm_measured:.4f}s")
    print(f"Speedup (legacy / fast_cold): {speedup_cold:.2f}x")
    print(f"Speedup (legacy / fast_warm): {speedup_warm:.2f}x")


if __name__ == '__main__':
    main()
