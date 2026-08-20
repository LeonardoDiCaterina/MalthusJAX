#!/usr/bin/env python3
"""Comprehensive High-Performance GPU Benchmark Suite for MalthusJAX.

Compares:
1. MalthusJAX Standard Engine (GeneticEngine)
2. MalthusJAX Lightened Engine (LightenedGeneticEngine - Fast-Path Monolithic)
3. MalthusJAX Batched Vectorized Engine (LightenedGeneticEngine + Whole-Array Tensor Ops)
4. EvoSAX (SimpleGA) via Universal Adapter (Pure JIT scan loop)

Supports CPU, Apple Silicon Metal (MPS), and NVIDIA CUDA GPUs.
Strictly enforces JAX device synchronization (`block_until_ready()`).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from malthusjax.engine.genetic_lightened import LightenedGeneticEngine

from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover import UniformCrossover
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.operators.selection import TournamentSelection


# ===========================================================================
# Benchmark Objective Functions
# ===========================================================================
class SphereEvaluator(BaseEvaluator):
    """Vectorized N-Dimensional Sphere Optimization Problem."""

    def evaluate_population(self, population: Any) -> Any:
        genes = getattr(population.genes, "values", population.genes)
        fitness = jnp.sum(jnp.square(genes), axis=-1)
        return population.replace(fitness=fitness)


# ===========================================================================
# Benchmark Suite Harness
# ===========================================================================
def run_benchmark_suite(
    pop_sizes: List[int],
    generations_list: List[int],
    dim: int = 10,
    seeds: Tuple[int, ...] = (42, 43),
    output_json: str = "gpu_benchmark_results.json",
) -> Dict[str, Any]:
    backend = jax.default_backend()
    devices = jax.devices()
    print("=" * 90)
    print("  MALTHUSJAX HIGH-PERFORMANCE GPU BENCHMARK SUITE")
    print(f"  JAX Device Backend: {backend.upper()} ({devices})")
    print(f"  Genome Dimension:   {dim}D Sphere Problem")
    print(f"  Population Sizes:   {pop_sizes}")
    print(f"  Generations:        {generations_list}")
    print("=" * 90)

    evaluator = SphereEvaluator(config=BaseEvaluatorConfig(maximize=False), data=None)

    results: Dict[str, Any] = {
        "backend": backend,
        "devices": [str(d) for d in devices],
        "dimension": dim,
        "experiments": [],
    }

    for pop_size in pop_sizes:
        for num_gens in generations_list:
            print(
                f"\n[Benchmarking Configuration] Pop Size: {pop_size:4d} | Generations: {num_gens:4d}"
            )
            print("-" * 85)

            genome_config = RealGenomeConfig(bounds=(-5.0, 5.0), shape=(dim,))
            params = GeneticEngineParams(pop_size=pop_size, num_generations=num_gens, elitism=0)

            # ---------------------------------------------------------------
            # 1. Standard GeneticEngine
            # ---------------------------------------------------------------
            std_engine = GeneticEngine(
                genome_config=genome_config,
                evaluator=evaluator,
                selection=TournamentSelection(num_selections=pop_size * 2, tournament_size=2),
                crossover=UniformCrossover(),
                mutation=GaussianMutation(mutation_rate=0.1, mutation_strength=0.1),
                engine_params=params,
            )

            # Warmup / JIT Compile
            k1 = jax.random.PRNGKey(seeds[0])
            std_state = std_engine.init_state(k1)
            std_compiled = jax.jit(std_engine.run).lower(std_state).compile()
            _ = std_compiled(std_state)

            std_times = []
            for s in seeds:
                st = std_engine.init_state(jax.random.PRNGKey(s))
                t0 = time.perf_counter()
                res, _, _ = std_compiled(st)
                res.best_fitness.block_until_ready()
                t1 = time.perf_counter()
                std_times.append(t1 - t0)

            std_mean_sec = float(np.mean(std_times))
            std_ms_per_gen = (std_mean_sec / num_gens) * 1000

            # ---------------------------------------------------------------
            # 2. LightenedGeneticEngine (Monolithic Fast-Path)
            # ---------------------------------------------------------------
            light_engine = LightenedGeneticEngine(
                genome_config=genome_config,
                evaluator=evaluator,
                selection=TournamentSelection(num_selections=pop_size * 2, tournament_size=2),
                crossover=UniformCrossover(),
                mutation=GaussianMutation(mutation_rate=0.1, mutation_strength=0.1),
                engine_params=params,
                use_vectorized_operators=False,
            )

            light_state = light_engine.init_state(k1)
            light_compiled = jax.jit(light_engine.run).lower(light_state).compile()
            _ = light_compiled(light_state)

            light_times = []
            for s in seeds:
                lst = light_engine.init_state(jax.random.PRNGKey(s))
                t0 = time.perf_counter()
                res, _, _ = light_compiled(lst)
                res.best_fitness.block_until_ready()
                t1 = time.perf_counter()
                light_times.append(t1 - t0)

            light_mean_sec = float(np.mean(light_times))
            light_ms_per_gen = (light_mean_sec / num_gens) * 1000

            # ---------------------------------------------------------------
            # 3. LightenedGeneticEngine with Whole-Tensor Batched Operators
            # ---------------------------------------------------------------
            batch_engine = LightenedGeneticEngine(
                genome_config=genome_config,
                evaluator=evaluator,
                selection=TournamentSelection(num_selections=pop_size * 2, tournament_size=2),
                crossover=UniformCrossover(),
                mutation=GaussianMutation(mutation_rate=0.1, mutation_strength=0.1),
                engine_params=params,
                use_vectorized_operators=True,
            )

            batch_state = batch_engine.init_state(k1)
            batch_compiled = jax.jit(batch_engine.run).lower(batch_state).compile()
            _ = batch_compiled(batch_state)

            batch_times = []
            for s in seeds:
                bst = batch_engine.init_state(jax.random.PRNGKey(s))
                t0 = time.perf_counter()
                res, _, _ = batch_compiled(bst)
                res.best_fitness.block_until_ready()
                t1 = time.perf_counter()
                batch_times.append(t1 - t0)

            batch_mean_sec = float(np.mean(batch_times))
            batch_ms_per_gen = (batch_mean_sec / num_gens) * 1000

            # ---------------------------------------------------------------
            # 4. EvoSAX (SimpleGA) via Universal Adapter (Pure JIT Scan)
            # ---------------------------------------------------------------
            evosax_mean_sec = None
            evosax_ms_per_gen = None
            try:
                evo_adapter = build_evosax_engine(
                    strategy_name="SimpleGA",
                    evaluator=evaluator,
                    pop_size=pop_size,
                    generations=num_gens,
                    bounds=(-5.0, 5.0),
                )
                evo_state = evo_adapter.init_state(k1)
                evo_compiled = jax.jit(evo_adapter.run).lower(evo_state).compile()
                _ = evo_compiled(evo_state)

                evo_times = []
                for s in seeds:
                    est = evo_adapter.init_state(jax.random.PRNGKey(s))
                    t0 = time.perf_counter()
                    res, _, _ = evo_compiled(est)
                    res.best_fitness.block_until_ready()
                    t1 = time.perf_counter()
                    evo_times.append(t1 - t0)

                evosax_mean_sec = float(np.mean(evo_times))
                evosax_ms_per_gen = (evosax_mean_sec / num_gens) * 1000
            except Exception:
                pass

            # ---------------------------------------------------------------
            # Summary Output Formatting
            # ---------------------------------------------------------------
            speedup_light = std_mean_sec / light_mean_sec if light_mean_sec > 0 else 0.0
            speedup_batch = std_mean_sec / batch_mean_sec if batch_mean_sec > 0 else 0.0

            print(
                f"  1. Standard GeneticEngine     | Total: {std_mean_sec * 1000:7.2f} ms | Per Gen: {std_ms_per_gen * 1000:6.1f} µs"
            )
            print(
                f"  2. LightenedEngine (Modular)  | Total: {light_mean_sec * 1000:7.2f} ms | Per Gen: {light_ms_per_gen * 1000:6.1f} µs (Speedup: {speedup_light:.2f}x)"
            )
            print(
                f"  3. Batched Vectorized Engine  | Total: {batch_mean_sec * 1000:7.2f} ms | Per Gen: {batch_ms_per_gen * 1000:6.1f} µs (Speedup: {speedup_batch:.2f}x)"
            )
            if evosax_ms_per_gen is not None and evosax_mean_sec is not None:
                print(
                    f"  4. EvoSAX (SimpleGA Adapter)  | Total: {evosax_mean_sec * 1000:7.2f} ms | Per Gen: {evosax_ms_per_gen * 1000:6.1f} µs"
                )

            exp_entry = {
                "pop_size": pop_size,
                "num_generations": num_gens,
                "standard_engine": {
                    "total_ms": std_mean_sec * 1000,
                    "us_per_gen": std_ms_per_gen * 1000,
                },
                "lightened_engine": {
                    "total_ms": light_mean_sec * 1000,
                    "us_per_gen": light_ms_per_gen * 1000,
                    "speedup_vs_standard": speedup_light,
                },
                "batched_vectorized_engine": {
                    "total_ms": batch_mean_sec * 1000,
                    "us_per_gen": batch_ms_per_gen * 1000,
                    "speedup_vs_standard": speedup_batch,
                },
                "evosax_simple_ga": {
                    "total_ms": evosax_mean_sec * 1000 if evosax_mean_sec else None,
                    "us_per_gen": evosax_ms_per_gen * 1000 if evosax_ms_per_gen else None,
                },
            }
            results["experiments"].append(exp_entry)

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print(f"  GPU Benchmark Suite Finished! Results saved to: {output_json}")
    print("=" * 90)
    return results


# ===========================================================================
# Entry point CLI
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="MalthusJAX High-Performance GPU Benchmark Suite")
    parser.add_argument(
        "--pop-sizes", type=int, nargs="+", default=[128, 512, 2048], help="Population sizes"
    )
    parser.add_argument("--generations", type=int, nargs="+", default=[50, 200], help="Generations")
    parser.add_argument("--dim", type=int, default=10, help="Genome problem dimension")
    parser.add_argument(
        "--output", type=str, default="gpu_benchmark_results.json", help="Output JSON path"
    )
    args = parser.parse_args()

    run_benchmark_suite(
        pop_sizes=args.pop_sizes,
        generations_list=args.generations,
        dim=args.dim,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
