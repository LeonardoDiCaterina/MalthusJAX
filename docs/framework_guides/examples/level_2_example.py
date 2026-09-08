"""Level 2 Example: Custom Operators with the Composable Evaluator Stack.

This script demonstrates Level 2: adding structured genetic operators
(Mutation, Crossover, Selection) on top of the Level 1 building blocks.

Key things demonstrated:
  - Define a custom Mutation following the 3-Tier Architecture.
  - Wire the fitness function with the composable evaluator stack
    (SphereEnv × IdentityTransform × IdentityInterpreter × ScalarOutput).
  - Run a manual lax.scan loop using the structured operator API —
    no Engine, no Resource Mapper — but with proper PRNG budgeting.
"""

import time
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

# Level 1 Core
from malthusjax.core.base import BaseGenome, BasePopulation

# Composable evaluator stack (Level 1 new paradigm)
from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from malthusjax.core.fitness.composable.environments import SphereEnv
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

# Level 2 Operators
from malthusjax.operators.base import BaseMutation


# ==============================================================================
# 1. Define the Genome (Level 1)
# ==============================================================================

@struct.dataclass
class ContinuousGenomeConfig:
    shape: Tuple[int, ...] = struct.field(pytree_node=False)

    def init_population(self, key: chex.PRNGKey, size: int) -> "ContinuousPopulation":
        return ContinuousPopulation.init_random(key, self, size)


@struct.dataclass
class ContinuousGenome(BaseGenome):
    values: chex.Array


@struct.dataclass
class ContinuousPopulation(BasePopulation[ContinuousGenome]):
    genes: ContinuousGenome
    fitness: chex.Array
    config: ContinuousGenomeConfig = struct.field(pytree_node=False)

    @classmethod
    def init_random(
        cls, key: chex.PRNGKey, config: ContinuousGenomeConfig, size: int
    ) -> "ContinuousPopulation":
        keys = jax.random.split(key, size)

        def init_one(k: chex.PRNGKey) -> ContinuousGenome:
            return ContinuousGenome(
                values=jax.random.uniform(k, shape=config.shape, minval=-5.12, maxval=5.12)
            )

        return cls(
            genes=jax.vmap(init_one)(keys),
            fitness=jnp.full((size,), jnp.inf),
            config=config,
        )


# ==============================================================================
# 2. Compose the Evaluator (Level 1 new paradigm)
#
# No subclassing needed. The fitness function is defined by composing four axes:
#   Environment   SphereEnv         → problem: minimize sum(x^2)
#   Transform     IdentityTransform  → no genotype-phenotype mapping
#   Interpreter   IdentityInterpreter→ pass genome values through as solution
#   Output        ScalarOutput       → single scalar, minimization convention
# ==============================================================================

evaluator = OptimizationEvaluator(
    env=SphereEnv(),
    transform=IdentityTransform(),
    interpreter=IdentityInterpreter(),
    output=ScalarOutput(maximize=False),
)


# ==============================================================================
# 3. Define a Custom Level 2 Operator (3-Tier Architecture)
#
# The 3-Tier separation is enforced by the base class:
#   Tier 1 (_mutate_one)     : pure math — no PRNG keys
#   Tier 2 (_generate_noise) : pure randomness — no genomes
#   Tier 3 (__call__)        : provided by BaseMutation — no hand-rolling vmap!
# ==============================================================================

@struct.dataclass
class ContinuousGaussianMutation(BaseMutation[ContinuousGenome, ContinuousGenomeConfig]):
    """Gaussian noise mutation following the 3-Tier architecture."""

    mutation_rate: float = struct.field(pytree_node=False, default=0.1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # One key per individual to generate the noise vector
        return 1

    def _generate_noise(
        self, keys: chex.Array, config: ContinuousGenomeConfig, generation: int = 0
    ) -> Tuple[chex.Array]:
        """Tier 2: Pure Randomness (No Genomes Allowed)."""
        noise = jax.random.normal(keys[0], shape=config.shape)
        return (noise,)

    def _mutate_one(
        self,
        genome: ContinuousGenome,
        noise_data: Tuple[chex.Array],
        config: ContinuousGenomeConfig,
        **kwargs: Any,
    ) -> ContinuousGenome:
        """Tier 1: Pure Math (No PRNG Keys Allowed)."""
        (noise,) = noise_data
        return genome.replace(values=genome.values + noise * self.mutation_rate)

    # Tier 3 (__call__) is automatically provided by BaseMutation!


# ==============================================================================
# 4. The Custom Loop (Level 1 + Level 2, no Engine)
# ==============================================================================

def run_level_2_evolution():
    POP_SIZE = 128
    NUM_GENERATIONS = 100

    config = ContinuousGenomeConfig(shape=(10,))
    mutator = ContinuousGaussianMutation(mutation_rate=0.1)

    # Initialize population and evaluate using the composable evaluator
    master_key = jax.random.PRNGKey(42)
    k_init, master_key = jax.random.split(master_key)
    population = config.init_population(k_init, POP_SIZE)
    population = evaluator.evaluate_population(population)

    @jax.jit
    def step(
        state: Tuple[chex.PRNGKey, ContinuousPopulation], _
    ) -> Tuple[Tuple[chex.PRNGKey, ContinuousPopulation], chex.Array]:
        key, pop = state
        k_mut, next_key = jax.random.split(key)

        # Level 2 key budgeting: ask the operator how many keys it needs
        expected_keys = mutator.num_keys((POP_SIZE,))
        mut_keys = jax.random.split(k_mut, expected_keys)

        # Tier 3 __call__ handles the vmap internally — no manual vmap!
        mutated_pop = mutator(mut_keys, pop, config)

        # Evaluate with the composable evaluator (deterministic → no rng needed)
        mutated_pop = evaluator.evaluate_population(mutated_pop)

        # Elitist selection
        improved = mutated_pop.fitness < pop.fitness
        final_values = jnp.where(
            improved[:, None], mutated_pop.genes.values, pop.genes.values
        )
        final_fitness = jnp.where(improved, mutated_pop.fitness, pop.fitness)

        final_pop = pop.replace(
            genes=ContinuousGenome(values=final_values),
            fitness=final_fitness,
        )
        return (next_key, final_pop), jnp.min(final_pop.fitness)

    t0 = time.time()
    (_, final_pop), history = jax.lax.scan(
        step, (master_key, population), None, length=NUM_GENERATIONS
    )
    t1 = time.time()

    print(f"Level 2 Evolution completed in {t1 - t0:.4f}s")
    print(f"Best fitness (last 5 gens): {history[-5:]}")
    print(f"Final best fitness: {jnp.min(final_pop.fitness):.6f}")


if __name__ == "__main__":
    run_level_2_evolution()
