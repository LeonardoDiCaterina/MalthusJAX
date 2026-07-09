import time
import jax
import jax.numpy as jnp
import chex
from flax import struct
from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealGenome
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.island_model.topologies import RingTopologyIsland, FullyConnectedIsland
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import PolynomialMutation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
import matplotlib.pyplot as plt

# --- Problem Definition: Rastrigin Function ---
# Rastrigin is a highly multi-modal function often used to test global optimization algorithms.
def rastrigin(x: jax.Array) -> jax.Array:
    # x shape: (..., n_dims)
    A = 10.0
    n = x.shape[-1]
    # Minimize, so return negative for maximization engine
    val = A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x), axis=-1)
    return -val

@struct.dataclass
class RastriginEvaluator(BaseEvaluator[RealGenome, BaseEvaluatorConfig, type(None)]):
    def evaluate(self, genes: RealGenome) -> jax.Array:
        return rastrigin(genes.values)

def run_island_model_demo():
    print("Initializing Asynchronous Island Model Demo...")
    
    # 1. Configuration
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    dim = 20
    pop_size_per_island = 250
    num_islands = 4
    total_pop_size = pop_size_per_island * num_islands
    num_generations = 2000
    migration_interval = 50
    num_migrants = 10
    
    config = RealGenomeConfig(shape=(dim,), bounds=(-5.12, 5.12))
    
    # 2. Base Engine Configuration
    params = GeneticEngineParams(
        pop_size=pop_size_per_island,
        elitism=5,
    )
    
    crossover = SimulatedBinaryCrossover(crossover_rate=0.9, eta=15.0)
    mutation = PolynomialMutation(mutation_rate=1.0/dim, eta=20.0)
    
    selection_island = TournamentSelection(num_selections=pop_size_per_island, tournament_size=3)
    base_engine = GeneticEngine(
        genome_config=config,
        evaluator=RastriginEvaluator(config=BaseEvaluatorConfig(maximize=True), data=None),
        selection=selection_island,
        crossover=crossover,
        mutation=mutation,
        engine_params=params,
    )
    
    # 3. Create Island Models
    ring_island = RingTopologyIsland(
        engine=base_engine,
        num_islands=num_islands,
        migration_interval=migration_interval,
        num_migrants=num_migrants
    )
    
    fully_connected_island = FullyConnectedIsland(
        engine=base_engine,
        num_islands=num_islands,
        migration_interval=migration_interval,
        num_migrants=num_migrants
    )
    
    # Baseline non-island model (equivalent total pop size)
    selection_baseline = TournamentSelection(num_selections=total_pop_size, tournament_size=3)
    baseline_params = GeneticEngineParams(pop_size=total_pop_size, elitism=5*num_islands)
    baseline_engine = GeneticEngine(
        genome_config=config,
        evaluator=RastriginEvaluator(config=BaseEvaluatorConfig(maximize=True), data=None),
        selection=selection_baseline,
        crossover=crossover,
        mutation=mutation,
        engine_params=baseline_params,
    )
    
    def run_model(model, init_fn, step_fn, gens, key):
        t0 = time.time()
        key, subkey = jax.random.split(key)
        
        # Initialize state
        state = init_fn(subkey)
            
        @jax.jit
        def train_loop(carry_state, _):
            next_state, history_output = step_fn(carry_state)
            
            # Find the best fitness
            # Island model returns history (migration_interval) x num_islands
            # Baseline returns single generation KPI
            return next_state, history_output
            
        final_state, history = jax.lax.scan(
            train_loop, 
            state, 
            None,
            length=gens
        )
        
        # Wait for compilation and execution
        history = jax.block_until_ready(history)
        t1 = time.time()
        print(f"Completed in {t1-t0:.2f}s")
        return history
    
    print("\\nRunning Baseline (Monolithic Population) -> 1000 Individuals")
    baseline_history_raw = run_model(baseline_engine, baseline_engine.init_state, baseline_engine.step, num_generations, key)
    baseline_history = baseline_history_raw.best_fitness
    print(f"Final Best Fitness (Baseline): {baseline_history[-1]:.4f}")
    
    print(f"\\nRunning Ring Topology Island -> {num_islands} Islands x {pop_size_per_island} Individuals")
    ring_history_raw = run_model(ring_island, ring_island.init_state, ring_island.step, num_generations // migration_interval, key)
    # ring_history_raw.best_fitness shape: (outer_gens, num_islands, migration_interval)
    # We max over islands, then flatten
    ring_history = jnp.max(ring_history_raw.best_fitness, axis=1).flatten()
    print(f"Final Best Fitness (Ring): {ring_history[-1]:.4f}")
    
    print(f"\\nRunning Fully Connected Island -> {num_islands} Islands x {pop_size_per_island} Individuals")
    fc_history_raw = run_model(fully_connected_island, fully_connected_island.init_state, fully_connected_island.step, num_generations // migration_interval, key)
    fc_history = jnp.max(fc_history_raw.best_fitness, axis=1).flatten()
    print(f"Final Best Fitness (Fully Connected): {fc_history[-1]:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_history, label="Baseline (1x1000)", alpha=0.8)
    plt.plot(ring_history, label="Ring Island (4x250)", alpha=0.8)
    plt.plot(fc_history, label="Fully Connected Island (4x250)", alpha=0.8)
    plt.title("Rastrigin Optimization (20D): Island Models vs Monolithic")
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness (Higher is Better, Max=0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("examples/island_model_comparison.png")
    print("\\nPlot saved to examples/island_model_comparison.png")

if __name__ == "__main__":
    run_island_model_demo()
