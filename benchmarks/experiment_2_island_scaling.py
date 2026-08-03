"""
Experiment 2: Island Model Scaling (Architecture)

Benchmarks the compilation and execution scaling of MalthusJAX Island Model
as the number of parallel demes increases.
"""
import time
import jax
import pandas as pd

from malthusjax.core.fitness.bbobax_evaluator import BBOBAXConfig, BBOBAXEvaluator
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealGenome
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import PolynomialMutation
from malthusjax.engine.island_model.topologies import RingTopologyIsland

def run_scaling_benchmark():
    dims = 20
    pop_size = 128
    num_gens = 500
    island_counts = [4, 16, 64]
    
    evaluator = BBOBAXEvaluator.create(
        BBOBAXConfig(fn_name="rastrigin", num_dims=dims, seed=42, maximize=False)
    )
    
    base_params = GeneticEngineParams(pop_size=pop_size, num_generations=num_gens)
    genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))
    base_engine = GeneticEngine(
        engine_params=base_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=TournamentSelection(tournament_size=3, num_selections=pop_size),
        crossover=SimulatedBinaryCrossover(eta=15.0, crossover_rate=0.9),
        mutation=PolynomialMutation(eta=15.0, mutation_rate=0.1, clip=True)
    )
    
    results = []
    
    for num_islands in island_counts:
        print(f"\nEvaluating Island Model with {num_islands} Islands...")
        
        island_engine = RingTopologyIsland(
            engine=base_engine,
            num_islands=num_islands,
            migration_interval=50,
            num_migrants=5
        )
        
        key = jax.random.PRNGKey(42)
        
        def run_fn(rng):
            init_key, loop_key = jax.random.split(rng)
            state = island_engine.init_state(init_key)
            def step_fn(carry, _):
                next_state, metrics = island_engine.step(carry)
                return next_state, metrics
            final_state, history = jax.lax.scan(step_fn, state, None, length=num_gens)
            return final_state, history
        
        # Compile
        t0 = time.perf_counter()
        compiled_fn = jax.jit(run_fn).lower(key).compile()
        t1 = time.perf_counter()
        compile_time = t1 - t0
        
        # Execute
        t2 = time.perf_counter()
        res = compiled_fn(key)
        jax.block_until_ready(res)
        t3 = time.perf_counter()
        exec_time = t3 - t2
        
        final_state, _ = res
        # best_fitness in Island Model is shaped (num_islands,)
        best_fit = float(final_state.best_fitness.min())
        
        results.append({
            "Num Islands": num_islands,
            "Total Population": num_islands * pop_size,
            "Compile Time (s)": compile_time,
            "Execution Time (s)": exec_time,
            "Throughput (islands/s)": (num_gens * num_islands) / exec_time,
            "Global Best Fitness": best_fit
        })
        
        print(f"  Compile Time: {compile_time:.2f}s")
        print(f"  Execution Time: {exec_time:.2f}s")
        print(f"  Global Best Fitness: {best_fit:.2f}")
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("\n--- Running Island Model Scaling Benchmark ---")
    df = run_scaling_benchmark()
    
    print("\n=== Experiment 2 Results ===")
    print(df.to_string())
    df.to_csv("benchmarks/experiment_2_results.csv", index=False)
