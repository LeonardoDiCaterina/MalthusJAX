"""Level 1 Example: Manual Evolution with the Composable Evaluator Stack.

This script demonstrates how to use MalthusJAX at its lowest level:
  - Define a custom genome (RealGenome-compatible).
  - Wire up a fitness function using the 4-axis composable evaluator:
      Environment (SphereEnv)  ×  Transform (IdentityTransform)
      × Interpreter (IdentityInterpreter)  ×  Output (ScalarOutput)
      → OptimizationEvaluator
  - Write your own lax.scan evolution loop — no Engine, no Resource Mapper.

The composable evaluator is fully JIT-compatible and works without the Engine.
"""

import time
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

# -- Composable evaluator stack (the new Level 1 paradigm) --------------------
from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from malthusjax.core.fitness.composable.environments import SphereEnv
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter


# ==============================================================================
# 1. Define the Genome (Struct-of-Arrays)
#
# At Level 1 you define your own genome type. Here we use a simple real-valued
# vector, conceptually identical to RealGenome but written out by hand to show
# exactly what is happening inside the framework.
# ==============================================================================

@struct.dataclass
class ContinuousGenomeConfig:
    """Static (non-pytree) configuration for our genome."""
    shape: Tuple[int, ...] = struct.field(pytree_node=False)

    def init_population(self, key: chex.PRNGKey, size: int) -> "ContinuousPopulation":
        return ContinuousPopulation.init_random(key, self, size)


@struct.dataclass
class ContinuousGenome(BaseGenome):
    """A pure JAX PyTree representing a continuous real-valued genome."""
    values: chex.Array  # Shape: (dim,)


@struct.dataclass
class ContinuousPopulation(BasePopulation[ContinuousGenome]):
    """Lifted Struct-of-Arrays population.

    ``genes.values`` has shape ``(pop_size, dim)`` — not a list of genomes,
    but a single genome with a leading batch dimension.
    """
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

        # Level 1 magic: vmap over the per-individual init to build the population
        batched_genes = jax.vmap(init_one)(keys)
        initial_fitness = jnp.full((size,), jnp.inf)  # inf = unseen (minimizing)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)


# ==============================================================================
# 2. Wire up the Composable Evaluator
#
# The new paradigm decomposes fitness into four independent axes:
#
#   Environment   – what is the problem?   (SphereEnv: minimize sum(x^2))
#   Transform     – genotype → phenotype?  (IdentityTransform: no-op)
#   Interpreter   – genome → output?       (IdentityInterpreter: pass values through)
#   Output        – how to aggregate?      (ScalarOutput: single scalar, minimization)
#
# These plug into OptimizationEvaluator, which handles vmap internally.
# ==============================================================================

evaluator = OptimizationEvaluator(
    env=SphereEnv(),
    transform=IdentityTransform(),
    interpreter=IdentityInterpreter(),
    output=ScalarOutput(maximize=False),
)


# ==============================================================================
# 3. The Custom Loop (No Engine, No Resource Mapper)
# ==============================================================================

def run_level_1_evolution():
    POP_SIZE = 128
    NUM_GENERATIONS = 100
    MUTATION_RATE = 0.1

    config = ContinuousGenomeConfig(shape=(10,))

    # --- Initialization ---
    master_key = jax.random.PRNGKey(42)
    k_init, master_key = jax.random.split(master_key)

    population = config.init_population(k_init, POP_SIZE)

    # Composable evaluator's evaluate_population replaces the old
    # dispatch_evaluate_population call — same effect, new paradigm.
    population = evaluator.evaluate_population(population)

    # --- Manual mutation (pure JAX, explicitly vmapped) ---
    def manual_mutate(genome: ContinuousGenome, key: chex.PRNGKey) -> ContinuousGenome:
        noise = jax.random.normal(key, shape=genome.values.shape)
        return genome.replace(values=genome.values + noise * MUTATION_RATE)

    batched_mutate = jax.vmap(manual_mutate, in_axes=(0, 0))

    # --- Single generation step (JIT-compiled) ---
    @jax.jit
    def step(
        state: Tuple[chex.PRNGKey, ContinuousPopulation], _
    ) -> Tuple[Tuple[chex.PRNGKey, ContinuousPopulation], chex.Array]:
        key, pop = state
        k_mut, next_key = jax.random.split(key)
        mut_keys = jax.random.split(k_mut, POP_SIZE)

        # Mutate
        mutated_genes = batched_mutate(pop.genes, mut_keys)
        mutated_pop = pop.replace(genes=mutated_genes)

        # Evaluate with the composable evaluator (deterministic → no rng needed)
        mutated_pop = evaluator.evaluate_population(mutated_pop)

        # Elitist selection: keep the individual if it improved
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

    # --- Execute with lax.scan ---
    t0 = time.time()
    (_, final_pop), history = jax.lax.scan(
        step, (master_key, population), None, length=NUM_GENERATIONS
    )
    t1 = time.time()

    print(f"Level 1 Evolution completed in {t1 - t0:.4f}s")
    print(f"Best fitness (last 5 gens): {history[-5:]}")
    print(f"Final best fitness: {jnp.min(final_pop.fitness):.6f}")


if __name__ == "__main__":
    run_level_1_evolution()
