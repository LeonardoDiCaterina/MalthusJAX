#!/usr/bin/env python3
"""
MalthusJAX Architectural Ablation Benchmark

This script performs a step-by-step ablation of MalthusJAX abstractions,
starting from a raw procedural EvoSAX ask/tell loop and building up to the
fully-featured MalthusJAX GeneticEngine.

It measures the exact execution overhead of each architectural feature by
compiling the evolution kernels identically and passing a dummy fitness function
to ensure purely apples-to-apples computational comparisons.
"""

import argparse
import time

import jax
import jax.numpy as jnp
from evosax.algorithms.population_based.simple_ga import SimpleGA

from malthusjax.composer.engine_catalog import EngineRegistry
from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator
from malthusjax.engine.schedules import TrackBest, TrackMetrics


# ─── 1. Dummy Evaluator (Apples-to-Apples Fitness) ────────────
class DummyEvaluator(BaseEvaluator):
    """A zero-cost fitness function (sum of squares) to isolate framework overhead."""

    def evaluate(self, genome):
        return jnp.sum(genome.values**2)

    def evaluate_population(self, pop: BasePopulation) -> BasePopulation:
        fit = jnp.sum(pop.genes.values**2, axis=-1)
        return pop.replace(fitness=fit)


def setup_engine(
    engine_type: str,
    pop_size: int,
    gens: int,
    track_metrics: TrackMetrics,
    mutation_op: str = "evosax_gaussian:mutation_strength=1.0",
):
    """Instantiate a MalthusJAX engine via the catalog."""
    registry = EngineRegistry()
    evaluator = DummyEvaluator(config=None, data=None)
    adapter = registry.get(
        engine_type,
        evaluator=evaluator,
        selection=f"evosax_mimic_selection:num_selections={pop_size},elite_k={int(pop_size * 0.16)}",
        crossover="evosax_uniform_crossover:crossover_rate=0.3",
        mutation=mutation_op,
        genome_type="real",
        pop_size=pop_size,
        generations=gens,
        genome_shape=(9,),
        bounds=(-5.0, 5.0),
        elitism=0,
        forward_presplit_keys=True,
        track_metrics=track_metrics,
        track_best=TrackBest.NONE,
    )
    return adapter.genetic_engine


def bench_engine(
    name: str, engine_instance, key: jax.Array, runs: int, gens: int, return_history: bool
) -> float:
    """
    Benchmarks a MalthusJAX engine.
    To avoid Python-side overhead and JIT recompilation in dynamic closure scopes,
    we explicitly wrap the engine's step function in a compiled jax.lax.scan block.
    """
    state = engine_instance.init_state(key)

    @jax.jit
    def run_fn(rng):
        def scan_step(carry, _):
            st, k = carry
            new_st, history = engine_instance.step(st.replace(rng_key=k))
            if not return_history:
                history = ()
            return (new_st, new_st.rng_key), history

        final_carry, history = jax.lax.scan(scan_step, (state, rng), jnp.arange(gens))
        return final_carry[0]

    # Warmup + JIT compile
    final_state = run_fn(key)
    final_state.best_fitness.block_until_ready()

    start = time.perf_counter()
    for i in range(runs):
        out_state = run_fn(jax.random.PRNGKey(i))
        out_state.best_fitness.block_until_ready()
    end = time.perf_counter()

    ms_per_run = (end - start) / runs * 1000
    print(f"  {name:42s} {ms_per_run:8.2f} ms")
    return ms_per_run


def bench_evosax(key: jax.Array, pop_size: int, gens: int, runs: int) -> float:
    """Benchmarks the raw EvoSAX ask/tell procedural loop."""
    init_solution = jnp.zeros(9)
    strategy = SimpleGA(population_size=pop_size, solution=init_solution)
    params = strategy.default_params.replace(crossover_rate=0.3)

    pop_init = jnp.zeros((pop_size, 9))
    fit_init = jnp.zeros(pop_size)
    state = strategy.init(key, pop_init, fit_init, params)

    @jax.jit
    def run_fn(rng, init_state):
        def scan_step(carry, _):
            st, k = carry
            k, k_ask, k_tell = jax.random.split(k, 3)
            x, st = strategy.ask(k_ask, st, params)
            # Dummy fitness matching MalthusJAX DummyEvaluator
            fit = jnp.sum(x**2, axis=-1)
            st, _ = strategy.tell(k_tell, x, fit, st, params)
            return (st, k), None

        (final_state, _), _ = jax.lax.scan(scan_step, (init_state, rng), jnp.arange(gens))
        return final_state

    # Warmup + JIT compile
    out = run_fn(key, state)
    out.best_fitness.block_until_ready()

    start = time.perf_counter()
    for i in range(runs):
        out = run_fn(jax.random.PRNGKey(i), state)
        out.best_fitness.block_until_ready()
    end = time.perf_counter()

    ms_per_run = (end - start) / runs * 1000
    print(f"  {'0. EvoSAX (Raw Ask/Tell Baseline)':42s} {ms_per_run:8.2f} ms")
    return ms_per_run


def run_ablation(pop_size: int, gens: int, runs: int):
    print("=" * 68)
    print("  MALTHUSJAX ARCHITECTURAL ABLATION STUDY")
    print(f"  D=9, Pop={pop_size}, Gens={gens}, Runs={runs} (Dummy Fitness)")
    print("=" * 68)

    key = jax.random.PRNGKey(42)

    t_evosax = bench_evosax(key, pop_size, gens, runs)

    # 1. NativeFastEngine (No Tracking/History)
    e_fast_none = setup_engine("native_fast", pop_size, gens, TrackMetrics.NONE)
    t_fast_none = bench_engine(
        "1. NativeFastEngine (No Tracking/History)", e_fast_none, key, runs, gens, False
    )

    # 2. NativeFastEngine (+ History Tensor)
    e_fast_hist = setup_engine("native_fast", pop_size, gens, TrackMetrics.NONE)
    t_fast_hist = bench_engine(
        "2. NativeFastEngine (+ History Tensor)", e_fast_hist, key, runs, gens, True
    )

    # 3. NativeFastEngine (+ Hist + Metrics)
    e_fast_metrics = setup_engine("native_fast", pop_size, gens, TrackMetrics.BASIC)
    t_fast_metrics = bench_engine(
        "3. NativeFastEngine (+ Hist + Metrics)", e_fast_metrics, key, runs, gens, True
    )

    # 4. GeneticEngine (No Tracking/History)
    e_gen_none = setup_engine("ga", pop_size, gens, TrackMetrics.NONE)
    bench_engine("4. GeneticEngine (No Tracking/History)", e_gen_none, key, runs, gens, False)

    # 5. GeneticEngine (Full Featured Default)
    e_gen_full = setup_engine("ga", pop_size, gens, TrackMetrics.BASIC)
    t_gen_full = bench_engine(
        "5. GeneticEngine (Full Featured Default)", e_gen_full, key, runs, gens, True
    )

    # 6. GeneticEngine (Batched EvoSAX Mutation)
    e_gen_batched = setup_engine(
        "ga",
        pop_size,
        gens,
        TrackMetrics.BASIC,
        mutation_op="batched_evosax_gaussian:mutation_strength=1.0",
    )
    t_gen_batched = bench_engine(
        "6. GeneticEngine (Batched EvoSAX Mutation)", e_gen_batched, key, runs, gens, True
    )

    print("=" * 68)
    print("  OVERHEAD ANALYSIS (relative to previous step)")
    print("=" * 68)
    print(f"  Base EvoSAX:                  {t_evosax:.2f} ms")
    print(
        f"  + MalthusJAX Abstractions:    {t_fast_none - t_evosax:+.2f} ms  (-> {t_fast_none:.2f} ms)"
    )
    print(
        f"  + History Tensor Allocation:  {t_fast_hist - t_fast_none:+.2f} ms  (-> {t_fast_hist:.2f} ms)"
    )
    print(
        f"  + Metric Math (Mean/Std):     {t_fast_metrics - t_fast_hist:+.2f} ms  (-> {t_fast_metrics:.2f} ms)"
    )
    print(
        f"  + PyTree passing in Scan:     {t_gen_full - t_fast_metrics:+.2f} ms  (-> {t_gen_full:.2f} ms)"
    )
    print(
        f"  + Batched EvoSAX Operator:    {t_gen_batched - t_gen_full:+.2f} ms  (-> {t_gen_batched:.2f} ms)"
    )
    print("=" * 68)

    total_overhead_default = t_gen_full - t_evosax
    percent_overhead_default = (total_overhead_default / t_evosax) * 100
    total_overhead_batched = t_gen_batched - t_evosax
    percent_overhead_batched = (total_overhead_batched / t_evosax) * 100
    print(
        f"  TOTAL OVERHEAD (Default):     {total_overhead_default:+.2f} ms ({percent_overhead_default:+.1f}%)"
    )
    print(
        f"  TOTAL OVERHEAD (Batched):     {total_overhead_batched:+.2f} ms ({percent_overhead_batched:+.1f}%)"
    )
    print("====================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MalthusJAX Architecture Ablation Benchmark")
    parser.add_argument("--pop-size", type=int, default=195, help="Population size")
    parser.add_argument("--gens", type=int, default=387, help="Generations per run")
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of times to run the full evolution for averaging",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    if args.smoke:
        args.gens = 10
        args.runs = 3

    run_ablation(args.pop_size, args.gens, args.runs)
