# %% [markdown]
# # Distributed Evolution with Island Models
#
# Welcome to the 6th showcase of MalthusJAX! In this notebook, we explore the **Native Distributed Island Models**.
#
# One of the most powerful features of JAX is `jax.vmap`, which vectorizes any function across a batch dimension. In MalthusJAX, we use this to vectorize **entire evolutionary engines** across multiple independent populations (Islands). This allows for massive scaling on GPUs/TPUs without the overhead of Python multiprocessing!
#
# In this demo, we will optimize the highly multimodal **BBOB Gallagher 101-me peaks** function using a `RingTopologyIsland` model.

# %%
import time

import jax
import jax.numpy as jnp

from malthusjax.core.fitness.bbobax_evaluator import BBOBAXConfig, BBOBAXEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.island_model.topologies import RingTopologyIsland
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import PolynomialMutation
from malthusjax.operators.selection.tournament import TournamentSelection

# %% [markdown]
# ## 1. Island Configuration
#
# We define our island properties. Here we instantiate 4 isolated islands, each with 250 individuals, that will migrate their best individuals every 50 generations.

# %%
dim = 20
pop_size_per_island = 250
num_islands = 4
total_pop_size = pop_size_per_island * num_islands
num_generations = 2000
migration_interval = 50
num_migrants = 10

genome_config = RealGenomeConfig(shape=(dim,), bounds=(-5.0, 5.0))

# The base engine that will run ON EACH ISLAND
params = GeneticEngineParams(
    pop_size=pop_size_per_island,
    elitism=5,
)

crossover = SimulatedBinaryCrossover(crossover_rate=0.9, eta=15.0)
mutation = PolynomialMutation(mutation_rate=1.0 / dim, eta=20.0, clip=True)

# BBOB Gallagher 101-me peaks (Highly multimodal)
bbob_config = BBOBAXConfig(fn_name="gallagher_101_me", num_dims=dim, maximize=False)
evaluator = BBOBAXEvaluator.create(bbob_config)

selection = TournamentSelection(num_selections=pop_size_per_island, tournament_size=3)

base_engine = GeneticEngine(
    genome_config=genome_config,
    params=params,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    evaluator=evaluator,
)

# %% [markdown]
# ## 2. Topology Construction
#
# We wrap our `base_engine` in a `RingTopologyIsland`. The topology handles `vmap`-ping the engine across the `num_islands` axis, and performing ring-structured migration swaps every `migration_interval`.

# %%
# Wrap the base engine inside the Island Meta-Engine
island_model = RingTopologyIsland(
    base_engine=base_engine,
    num_islands=num_islands,
    migration_interval=migration_interval,
    num_migrants=num_migrants,
)

# Initialize the global state (num_islands, pop_size, dim)
key = jax.random.PRNGKey(42)
print(f"Initializing {num_islands} Islands (Total Population: {total_pop_size})...")

start_init = time.time()
global_state = jax.jit(island_model.init)(key)
# Block until JIT compilation completes
global_state.generation.block_until_ready()
print(f"Init & JIT Compilation took: {time.time() - start_init:.2f}s")

# %% [markdown]
# ## 3. Evolution!
#
# Notice the shape of `global_state.best_fitness`. It has a batch dimension of `num_islands` (in this case, 4). Each island explores its own local optimal landscape, and shares its best migrants with its neighbors!

# %%
print(f"Initial Best Fitness per Island: {global_state.best_fitness}")


# We can step the island model normally. Under the hood, it is stepping all islands in parallel.
@jax.jit
def run_islands(state):
    return jax.lax.fori_loop(0, num_generations, lambda _, s: island_model.step(s)[0], state)


print(
    f"\nRunning {num_generations} Generations ({num_generations * total_pop_size:,} total evaluations)..."
)
start_evol = time.time()
final_state = run_islands(global_state)
final_state.generation.block_until_ready()
time_evol = time.time() - start_evol
print(f"Evolution Completed in {time_evol:.2f}s!")

# Print final results
print(f"\nFinal Best Fitness per Island: {final_state.best_fitness}")
print(f"Global Best Fitness: {jnp.max(final_state.best_fitness):.4f}")
print(f"Total Speed: {(num_generations * total_pop_size) / time_evol:,.0f} evaluations/second")

# %% [markdown]
# Because `RingTopologyIsland` handles memory routing on-device using permutation matrices, we never have to copy memory back to the CPU between migrations, leading to absolute massive throughput on GPUs/TPUs!
