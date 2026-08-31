"""
Hero Benchmark: MalthusJAX Single-Kernel Generational Loop Throughput

Demonstrates how MalthusJAX fuses an entire 1,000-generation evolutionary loop
into a single compiled XLA kernel via jax.lax.scan.
"""

import time

import jax
import jax.numpy as jnp

from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import PolynomialMutation
from malthusjax.operators.selection.tournament import TournamentSelection


def sphere_evaluator(values):
    """Vectorized Sphere function evaluation."""
    return -jnp.sum(jnp.square(values), axis=-1)


def main():
    pop_size = 2048
    num_generations = 1000
    genome_dim = 50

    print("=================================================================")
    print("           MalthusJAX Hero Benchmark (Single XLA Program)")
    print("=================================================================")
    print(f"Device: {jax.devices()[0].device_kind.upper()} ({jax.devices()[0]})")
    print(f"Population Size: {pop_size:,}")
    print(f"Genome Dimensions: {genome_dim}")
    print(f"Generations: {num_generations:,}")
    print(f"Total Evaluations per Run: {pop_size * num_generations:,}")
    print("-----------------------------------------------------------------")

    # 1. Setup Engine & Configs
    engine_params = GeneticEngineParams(pop_size=pop_size, num_generations=num_generations)
    genome_config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))

    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=sphere_evaluator,
        selection=TournamentSelection(tournament_size=3, num_selections=pop_size),
        crossover=SimulatedBinaryCrossover(eta=15.0, crossover_rate=0.9),
        mutation=PolynomialMutation(eta=15.0, mutation_rate=0.1, clip=True),
    )

    # 2. Define single-program generational loop
    def run_evolution(seed_key):
        init_key, loop_key = jax.random.split(seed_key)
        state = engine.init_state(init_key)

        def step_fn(carry_state, _):
            next_state, metrics = engine.step(carry_state)
            return next_state, metrics

        final_state, history = jax.lax.scan(step_fn, state, None, length=num_generations)
        return final_state, history

    key = jax.random.PRNGKey(42)

    # 3. Compilation / Warmup Phase
    print("Compiling generational loop into single XLA program...")
    t_comp_start = time.perf_counter()
    compiled_run = jax.jit(run_evolution).lower(key).compile()
    t_comp_end = time.perf_counter()
    compile_duration = t_comp_end - t_comp_start
    print(f"XLA Compilation Time: {compile_duration * 1000:.2f} ms")

    # 4. Execution Phase (Timed)
    print("\nExecuting 1,000 generations on accelerator...")
    t_exec_start = time.perf_counter()
    final_state, _ = compiled_run(key)
    # Ensure all asynchronous GPU kernels finish before stopping timer
    final_state.best_fitness.block_until_ready()
    t_exec_end = time.perf_counter()

    exec_duration = t_exec_end - t_exec_start
    total_evals = pop_size * num_generations
    gens_per_sec = num_generations / exec_duration
    evals_per_sec = total_evals / exec_duration

    print("-----------------------------------------------------------------")
    print(f"Execution Duration:  {exec_duration * 1000:.2f} ms")
    print(f"Throughput (Gens):   {gens_per_sec:,.2f} generations/sec")
    print(f"Throughput (Evals):  {evals_per_sec:,.2f} evaluations/sec")
    print(f"Best Fitness Found:  {float(final_state.best_fitness):.6f}")
    print("=================================================================")


if __name__ == "__main__":
    main()
