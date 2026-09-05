import time
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, dispatch_evaluate_population

# ==============================================================================
# 1. Define the Genome (Struct-of-Arrays)
# ==============================================================================
@struct.dataclass
class ContinuousGenomeConfig:
    """Static configuration for our genome."""
    shape: Tuple[int, ...] = struct.field(pytree_node=False)

    def init_population(self, key: chex.PRNGKey, size: int) -> "ContinuousPopulation":
        return ContinuousPopulation.init_random(key, self, size)

@struct.dataclass
class ContinuousGenome(BaseGenome):
    """A pure JAX PyTree representing a continuous genome."""
    values: chex.Array

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: ContinuousGenomeConfig) -> "ContinuousGenome":
        # Initialize an individual with values in [-5.12, 5.12]
        values = jax.random.uniform(key, shape=config.shape, minval=-5.12, maxval=5.12)
        return cls(values=values)

@struct.dataclass
class ContinuousPopulation(BasePopulation[ContinuousGenome]):
    """Lifted representation. 'genes' is a ContinuousGenome holding batched arrays."""
    genes: ContinuousGenome
    fitness: chex.Array
    config: ContinuousGenomeConfig = struct.field(pytree_node=False)

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: ContinuousGenomeConfig, size: int) -> "ContinuousPopulation":
        keys = jax.random.split(key, size)
        # Magic of Level 1: We manually vmap the single-genome init function!
        batched_genes = jax.vmap(ContinuousGenome.random_init, in_axes=(0, None))(keys, config)
        initial_fitness = jnp.full((size,), jnp.inf) # Using inf since we'll minimize
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)


# ==============================================================================
# 2. Define the Evaluator
# ==============================================================================
@struct.dataclass
class SphereEvaluatorConfig:
    maximize: bool = struct.field(pytree_node=False, default=False)

@struct.dataclass
class SphereEvaluator(BaseEvaluator[ContinuousGenome, ContinuousGenomeConfig, SphereEvaluatorConfig]):
    """Scores individuals based on the Sphere function (sum of squares)."""

    def evaluate(
        self, genome: ContinuousGenome, **kwargs: Any
    ) -> chex.Array:
        # Simple sphere function: sum(x^2)
        return jnp.sum(genome.values ** 2)


# ==============================================================================
# 3. The Custom Loop (No Engine, No Resource Mapper)
# ==============================================================================
def run_level_1_evolution():
    # Setup hyperparameters
    POP_SIZE = 128
    NUM_GENERATIONS = 100
    MUTATION_RATE = 0.1

    config = ContinuousGenomeConfig(shape=(10,))
    evaluator_config = SphereEvaluatorConfig(maximize=False)
    evaluator = SphereEvaluator(config=evaluator_config, data=None)

    # 1. Initialize
    master_key = jax.random.PRNGKey(42)
    k_init, master_key = jax.random.split(master_key)

    population = config.init_population(k_init, POP_SIZE)
    population = dispatch_evaluate_population(evaluator, population, k_init)

    # 2. Define a custom manual mutation (pure JAX)
    def manual_mutate(genome: ContinuousGenome, key: chex.PRNGKey) -> ContinuousGenome:
        noise = jax.random.normal(key, shape=genome.values.shape)
        new_values = genome.values + (noise * MUTATION_RATE)
        return genome.replace(values=new_values)

    # We explicitly vmap our custom mutation to handle the population
    batched_mutate = jax.vmap(manual_mutate, in_axes=(0, 0))

    # 3. Define the manual generation step
    @jax.jit
    def step(state: Tuple[chex.PRNGKey, ContinuousPopulation], _) -> Tuple[Tuple[chex.PRNGKey, ContinuousPopulation], chex.Array]:
        key, pop = state
        k_mut, k_eval, next_key = jax.random.split(key, 3)

        # We must manually generate the exact number of keys needed
        mut_keys = jax.random.split(k_mut, POP_SIZE)

        # Apply mutation
        mutated_genes = batched_mutate(pop.genes, mut_keys)
        mutated_pop = pop.replace(genes=mutated_genes)

        # Evaluate
        mutated_pop = dispatch_evaluate_population(evaluator, mutated_pop, k_eval)

        # Naive Selection (accept if better)
        # Since we are minimizing:
        improved = mutated_pop.fitness < pop.fitness

        # We manually construct the merge logic using jnp.where
        final_values = jnp.where(improved[:, None], mutated_pop.genes.values, pop.genes.values)
        final_fitness = jnp.where(improved, mutated_pop.fitness, pop.fitness)

        final_pop = pop.replace(
            genes=ContinuousGenome(values=final_values),
            fitness=final_fitness
        )

        best_fitness = jnp.min(final_pop.fitness)
        return (next_key, final_pop), best_fitness

    # 4. Execute using manual lax.scan
    t0 = time.time()
    (final_key, final_pop), history = jax.lax.scan(
        step,
        (master_key, population),
        None,
        length=NUM_GENERATIONS
    )
    t1 = time.time()

    print(f"Level 1 Evolution completed in {t1-t0:.4f} seconds!")
    print(f"Best fitness over time: {history[-5:]}")
    print(f"Final best individual fitness: {jnp.min(final_pop.fitness)}")

if __name__ == "__main__":
    run_level_1_evolution()
