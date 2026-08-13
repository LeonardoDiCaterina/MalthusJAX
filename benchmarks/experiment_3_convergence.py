"""
Experiment 3: Exotic Island Convergence (Algorithmic Power)

Benchmarks the convergence of a monolithic EvoSax CMA-ES vs an
heterogeneous MalthusJAX Island Model wrapping EvoSax CMA-ES instances.
"""

import time

import jax
import pandas as pd

from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.bbobax_evaluator import BBOBAXConfig, BBOBAXEvaluator
from malthusjax.engine.island_model.topologies import RingTopologyIsland


def run_convergence_benchmark():
    dims = 20
    pop_size = 128
    num_gens = 200
    num_islands = 4

    # Using Gallagher's Gaussian 101-me peaks (highly multimodal, requires exploration)
    evaluator = BBOBAXEvaluator.create(
        BBOBAXConfig(fn_name="gallagher_101_me", num_dims=dims, seed=42, maximize=False)
    )

    # 1. Monolithic CMA-ES (pop = 128 * 4 = 512 for fair comparison)
    print("\n--- Running Convergence Benchmark (Dims: 20) ---")
    monolithic_pop = pop_size * num_islands
    monolithic_cma = build_evosax_engine(
        strategy_name="CMA_ES",
        pop_size=monolithic_pop,
        num_dims=dims,
        num_generations=num_gens,
        maximize=False,
        evaluator=evaluator,
        strategy_kwargs={"elite_ratio": 0.5},
    )

    # 2. MalthusJAX Island wrapped CMA-ES
    base_cma = build_evosax_engine(
        strategy_name="CMA_ES",
        pop_size=pop_size,
        num_dims=dims,
        num_generations=num_gens,
        maximize=False,
        evaluator=evaluator,
        strategy_kwargs={"elite_ratio": 0.5},
    )

    island_engine = RingTopologyIsland(
        engine=base_cma, num_islands=num_islands, migration_interval=20, num_migrants=2
    )

    results = []

    print("Evaluating Monolithic EvoSax CMA-ES...")
    key = jax.random.PRNGKey(42)
    t0 = time.perf_counter()
    res_mono = monolithic_cma.run_once(key, compile=True)
    t1 = time.perf_counter()
    best_mono = float(res_mono["summary"]["best_fitness"])
    print(f"  Execution Time: {t1 - t0:.2f}s")
    print(f"  Best Fitness: {best_mono:.2f}")

    print("Evaluating MalthusJAX CMA-ES Ring Island...")

    def run_fn(rng):
        init_key, loop_key = jax.random.split(rng)
        state = island_engine.init_state(init_key)

        def step_fn(carry, _):
            next_state, metrics = island_engine.step(carry)
            return next_state, metrics

        final_state, history = jax.lax.scan(step_fn, state, None, length=num_gens)
        return final_state, history

    # JIT compile and execute
    t0 = time.perf_counter()
    compiled_fn = jax.jit(run_fn).lower(key).compile()
    t1 = time.perf_counter()
    compile_time = t1 - t0

    t2 = time.perf_counter()
    res_island = compiled_fn(key)
    jax.block_until_ready(res_island)
    t3 = time.perf_counter()
    exec_time = t3 - t2

    final_state, _ = res_island
    best_island = float(final_state.best_fitness.min())
    print(f"  Compile Time: {compile_time:.2f}s")
    print(f"  Execution Time: {exec_time:.2f}s")
    print(f"  Global Best Fitness: {best_island:.2f}")

    results.append(
        {
            "Algorithm": "Monolithic EvoSax CMA-ES",
            "Total Pop": monolithic_pop,
            "Best Fitness": best_mono,
        }
    )
    results.append(
        {
            "Algorithm": "MalthusJAX Ring Island (CMA-ES)",
            "Total Pop": monolithic_pop,
            "Best Fitness": best_island,
        }
    )

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_convergence_benchmark()

    print("\n=== Experiment 3 Results ===")
    print(df.to_string())
    df.to_csv("benchmarks/experiment_3_results.csv", index=False)
