"""
Experiment 1: Raw Throughput (Speed) Benchmark

Compares MalthusJAX Native GA against EvoSax SimpleGA on Rastrigin.
"""

import time

import jax
import pandas as pd

from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.bbobax_evaluator import BBOBAXConfig, BBOBAXEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import PolynomialMutation
from malthusjax.operators.selection.tournament import TournamentSelection


def run_benchmark(dims):
    print(f"\n--- Running Throughput Benchmark (Dims: {dims}) ---")

    pop_size = 1024
    num_gens = 1000

    # 1. Define Evaluator (Rastrigin)
    evaluator = BBOBAXEvaluator.create(
        BBOBAXConfig(fn_name="rastrigin", num_dims=dims, seed=42, maximize=False)
    )

    # 2. MalthusJAX Native GA
    mjx_params = GeneticEngineParams(pop_size=pop_size, num_generations=num_gens)
    mjx_genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))
    mjx_engine = GeneticEngine(
        engine_params=mjx_params,
        genome_config=mjx_genome_config,
        evaluator=evaluator,
        selection=TournamentSelection(tournament_size=3, num_selections=pop_size),
        crossover=SimulatedBinaryCrossover(eta=15.0, crossover_rate=0.9),
        mutation=PolynomialMutation(eta=15.0, mutation_rate=0.1, clip=True),
    )

    # 3. EvoSax SimpleGA
    evosax_engine = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evaluator,
        pop_size=pop_size,
        num_generations=num_gens,
        bounds=(-5.0, 5.0),
        maximize=False,
        seed=42,
    )

    results = []

    for name, engine in [("MalthusJAX Native GA", mjx_engine), ("EvoSax SimpleGA", evosax_engine)]:
        print(f"Evaluating {name}...")
        key = jax.random.PRNGKey(42)

        if name == "EvoSax SimpleGA":
            # EvoSax wrapper handles its own JIT and timings inside run_once
            res = engine.run_once(key, compile=True)
            compile_time = (
                res["summary"].get("timings", {}).get("warmup", 0.0)
            )  # wait run_once returns dict without timings if legacy?
            # actually we don't have timings inside summary, timings is at root level but wait run_once returns dict with 'summary'

            # just time it from outside for simplicity if we want
            # wait, UniversalAdapterEngine.run_once computes warmup/execution.
            # let's just use wall clock.
            pass

        else:

            def run_fn(rng):
                init_key, loop_key = jax.random.split(rng)
                state = engine.init_state(init_key)

                def step_fn(carry, _):
                    next_state, metrics = engine.step(carry)
                    return next_state, metrics

                final_state, history = jax.lax.scan(step_fn, state, None, length=num_gens)
                return final_state, history

        # JIT compile and execute
        t0 = time.perf_counter()
        if name == "EvoSax SimpleGA":
            # Just call run_once, it will JIT internally on the first call
            res = engine.run_once(key, compile=True)
            # The UniversalAdapterEngine doesn't return timing in standard output, let's just consider it all execution
            compile_time = 0.0  # Can't split cleanly
            exec_time = time.perf_counter() - t0
        else:
            compiled_fn = jax.jit(run_fn).lower(key).compile()
            t1 = time.perf_counter()
            compile_time = t1 - t0

            t2 = time.perf_counter()
            res = compiled_fn(key)
            jax.block_until_ready(res)
            t3 = time.perf_counter()
            exec_time = t3 - t2

        if name == "EvoSax SimpleGA":
            best_fit = float(res["summary"]["best_fitness"])
        else:
            final_state, _ = res
            best_fit = float(final_state.best_fitness)

        results.append(
            {
                "Algorithm": name,
                "Dims": dims,
                "Compile Time (s)": compile_time,
                "Execution Time (s)": exec_time,
                "Throughput (gens/s)": num_gens / exec_time,
                "Best Fitness": best_fit,
            }
        )

        print(f"  Compile Time: {compile_time:.2f}s")
        print(f"  Execution Time: {exec_time:.2f}s")
        print(f"  Throughput: {num_gens / exec_time:.0f} gens/s")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df1 = run_benchmark(100)
    df2 = run_benchmark(1000)

    final_df = pd.concat([df1, df2], ignore_index=True)
    print("\n=== Experiment 1 Results ===")
    print(final_df.to_string())

    final_df.to_csv("benchmarks/experiment_1_results.csv", index=False)
