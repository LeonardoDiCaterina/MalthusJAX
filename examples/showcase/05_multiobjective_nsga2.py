# %% [markdown]
# # Multi-Objective Evolution with NSGA-II
#
# Welcome to the 5th showcase of MalthusJAX! In this notebook, we explore the **Native Multi-Objective Engine** (`MOEngine`), which implements the NSGA-II paradigm (Non-dominated Sorting Genetic Algorithm II).
#
# MalthusJAX natively supports multi-objective evolution without relying on external adapters like EvoSAX or QDAX. By using the `MOPopulation` data structure, Pareto sorting and crowding distance computations are fused directly into the JAX `vmap` logic, allowing for blazingly fast $\mu+\lambda$ survival selection!
#
# ## 1. Setup & Multi-Objective Evaluator
#
# First, let's implement the classic **ZDT1** benchmark.
# ZDT1 is a bi-objective minimization problem. We will implement it by inheriting from `BaseMOEvaluator`.

# %%
import time

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams
from malthusjax.operators.mutation.real import PolynomialMutation


@struct.dataclass
class ZDT1Config(BaseEvaluatorConfig):
    num_dims: int = 30
    maximize: bool = False


class ZDT1Evaluator(BaseMOEvaluator):
    @classmethod
    def create(cls, config: ZDT1Config) -> "ZDT1Evaluator":
        return cls(config=config, data=None)

    def evaluate(self, genome) -> chex.Array:
        # Extract values and clip to valid domain
        X = getattr(genome, "values", genome)
        X = jnp.clip(X, 1e-6, 1.0)

        # ZDT1 equations
        f1 = X[0]
        g = 1.0 + 9.0 * jnp.sum(X[1:]) / (self.config.num_dims - 1)
        f2 = g * (1.0 - jnp.sqrt(f1 / g))

        # MalthusJAX maximizes fitness natively, so we negate minimization objectives
        return jnp.stack([-f1, -f2], axis=-1)


# %% [markdown]
# ## 2. Engine and Operator Configuration
#
# We'll use a `GeneticMutationEmitter` for pure variation, and plug it into the `MOEngine`. The `MOEngine` will automatically use non-dominated sorting and crowding distances to prune the population.

# %%
dim = 30
pop_size = 100
generations = 200

# 1. Genome & Engine Parameters
genome_config = RealGenomeConfig(shape=(dim,), bounds=(0.0, 1.0))
params = MOEngineParams(pop_size=pop_size)

# 2. Evaluator
evaluator = ZDT1Evaluator.create(ZDT1Config(num_dims=dim))

# 3. Operators & Emitters
mutation = PolynomialMutation(mutation_rate=1.0 / dim, eta=20.0, clip=True)

from malthusjax.operators.emitters.genetic import GeneticMutationEmitter

emitter = GeneticMutationEmitter(
    mutation=mutation, genome_config=genome_config, _batch_size=pop_size
)

# 4. Engine Assembly
engine = MOEngine(
    emitter=emitter,
    evaluator=evaluator,
    engine_params=params,
)

# %% [markdown]
# ## 3. JIT Compilation and Evolution Loop
#
# We'll manually step the engine using `jax.lax.fori_loop` to keep the evolution on-device and extremely fast!

# %%
key = jax.random.PRNGKey(42)
k_init, k_engine = jax.random.split(key)
initial_pop = genome_config.init_population(k_init, pop_size)

print("JIT Compiling MOEngine Initialization...")
start_t = time.time()
state = jax.jit(engine.init_state)(k_engine, initial_pop)
state.generation.block_until_ready()
print(f"Init took {time.time() - start_t:.2f}s")


@jax.jit
def run_generations(state):
    return jax.lax.fori_loop(0, generations, lambda _, s: engine.step(s)[0], state)


print(
    f"JIT Compiling & Running {generations} Generations ({pop_size * generations:,} evaluations)..."
)
start_t = time.time()
final_state = run_generations(state)
final_state.generation.block_until_ready()
print(f"Evolution took {time.time() - start_t:.2f}s")

# %% [markdown]
# ## 4. Visualizing the Pareto Front
#
# Let's plot the final population's fitness. Since we inverted the fitnesses to maximize them internally, we revert them here to plot the minimization front.

# %%
fitness = final_state.population.fitness

# Invert back to minimization for plotting
f1 = -fitness[:, 0]
f2 = -fitness[:, 1]

# Filter out non-dominated individuals (rank 0)
is_pareto = final_state.population.pareto_rank == 0
pareto_f1 = f1[is_pareto]
pareto_f2 = f2[is_pareto]

plt.figure(figsize=(10, 6))
# Plot all individuals
plt.scatter(f1, f2, color="lightgray", edgecolors="k", alpha=0.5, label="Dominated")
# Highlight the Pareto front
plt.scatter(
    pareto_f1, pareto_f2, color="red", edgecolors="k", alpha=0.9, label="Pareto Front (Rank 0)"
)

plt.title("NSGA-II on ZDT1 (Native MalthusJAX)")
plt.xlabel("Objective 1 (f1)")
plt.ylabel("Objective 2 (f2)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# %% [markdown]
# **Notice how beautifully the Red points map out the exact optimal Pareto front for ZDT1**, perfectly balancing both objectives natively on hardware!
