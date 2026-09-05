import time
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, dispatch_evaluate_population
from malthusjax.operators.base import BaseMutation

# ==============================================================================
# 1. Define the Genome & Evaluator (Level 1: The Core)
# ==============================================================================
@struct.dataclass
class ContinuousGenomeConfig:
    shape: Tuple[int, ...] = struct.field(pytree_node=False)

    def init_population(self, key: chex.PRNGKey, size: int) -> "ContinuousPopulation":
        return ContinuousPopulation.init_random(key, self, size)

@struct.dataclass
class ContinuousGenome(BaseGenome):
    values: chex.Array

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: ContinuousGenomeConfig) -> "ContinuousGenome":
        values = jax.random.uniform(key, shape=config.shape, minval=-5.12, maxval=5.12)
        return cls(values=values)

@struct.dataclass
class ContinuousPopulation(BasePopulation[ContinuousGenome]):
    genes: ContinuousGenome
    fitness: chex.Array
    config: ContinuousGenomeConfig = struct.field(pytree_node=False)

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: ContinuousGenomeConfig, size: int) -> "ContinuousPopulation":
        keys = jax.random.split(key, size)
        batched_genes = jax.vmap(ContinuousGenome.random_init, in_axes=(0, None))(keys, config)
        return cls(genes=batched_genes, fitness=jnp.full((size,), jnp.inf), config=config)

@struct.dataclass
class SphereEvaluatorConfig:
    maximize: bool = struct.field(pytree_node=False, default=False)

@struct.dataclass
class SphereEvaluator(BaseEvaluator[ContinuousGenome, ContinuousGenomeConfig, SphereEvaluatorConfig]):
    def evaluate(self, genome: ContinuousGenome, **kwargs: Any) -> chex.Array:
        return jnp.sum(genome.values ** 2)

# ==============================================================================
# 2. Define the Operator (Level 2: 3-Tier Architecture)
# ==============================================================================
@struct.dataclass
class ContinuousGaussianMutation(BaseMutation[ContinuousGenome, ContinuousGenomeConfig]):
    """A Level 2 Operator enforcing the 3-Tier separation of concerns."""

    mutation_rate: float = struct.field(pytree_node=False, default=0.1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # We only need 1 key to generate the gaussian noise matrix
        return 1

    def _generate_noise(
        self, keys: chex.Array, config: ContinuousGenomeConfig, generation: int = 0
    ) -> Tuple[chex.Array]:
        """Tier 2: Pure Randomness (No Genomes Allowed)"""
        # We expect 1 key per individual because num_keys_per_atomic_operation = 1
        # Keys shape: (1,)
        noise = jax.random.normal(keys[0], shape=config.shape)
        return (noise,)

    def _mutate_one(
        self,
        genome: ContinuousGenome,
        noise_data: Tuple[chex.Array],
        config: ContinuousGenomeConfig,
        **kwargs: Any,
    ) -> ContinuousGenome:
        """Tier 1: Pure Math (No PRNG Keys Allowed)"""
        noise = noise_data[0]
        new_values = genome.values + (noise * self.mutation_rate)
        return genome.replace(values=new_values)

    # Tier 3 (__call__) is automatically provided by the BaseMutation class!


# ==============================================================================
# 3. The Custom Loop (Level 1 + Level 2)
# ==============================================================================
def run_level_2_evolution():
    POP_SIZE = 128
    NUM_GENERATIONS = 100

    config = ContinuousGenomeConfig(shape=(10,))
    evaluator_config = SphereEvaluatorConfig(maximize=False)
    evaluator = SphereEvaluator(config=evaluator_config, data=None)

    # 1. Initialize Operator
    mutator = ContinuousGaussianMutation(mutation_rate=0.1)

    # 2. Initialize State
    master_key = jax.random.PRNGKey(42)
    k_init, master_key = jax.random.split(master_key)
    population = config.init_population(k_init, POP_SIZE)
    population = dispatch_evaluate_population(evaluator, population, k_init)

    # 3. Define the manual generation step
    @jax.jit
    def step(state: Tuple[chex.PRNGKey, ContinuousPopulation], _) -> Tuple[Tuple[chex.PRNGKey, ContinuousPopulation], chex.Array]:
        key, pop = state
        k_mut, k_eval, next_key = jax.random.split(key, 3)

        # We still have to manually budget the PRNG keys (no ResourceMapper!)
        # We ask the mutator how many total keys it needs for the batch
        expected_keys = mutator.num_keys((POP_SIZE,))
        mut_keys = jax.random.split(k_mut, expected_keys)

        # Apply mutation using the robust Tier 3 Operator __call__
        # Look ma, no manual vmap!
        mutated_pop = mutator(mut_keys, pop, config)

        # Evaluate
        mutated_pop = dispatch_evaluate_population(evaluator, mutated_pop, k_eval)

        # Naive Selection (accept if better)
        improved = mutated_pop.fitness < pop.fitness
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

    print(f"Level 2 Evolution completed in {t1-t0:.4f} seconds!")
    print(f"Best fitness over time: {history[-5:]}")
    print(f"Final best individual fitness: {jnp.min(final_pop.fitness)}")

if __name__ == "__main__":
    run_level_2_evolution()
