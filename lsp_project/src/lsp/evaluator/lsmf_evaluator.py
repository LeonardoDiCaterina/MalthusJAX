"""LSMF Evaluator Wrapper.

Implements the Memetic "Learn" step of the LSMF algorithm by running local
gradient descent on the continuous parameters (weights and biases) of the genome
using optax, before evaluating the final fitness.
"""

from typing import Any

import chex
import jax
import optax
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, StochasticEvaluator


@struct.dataclass
class LSMFEvaluator(StochasticEvaluator[Any, Any, Any]):
    """Memetic wrapper applying SGD/Adam to continuous parameters."""

    base_evaluator: BaseEvaluator = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    epochs: int = struct.field(pytree_node=False)

    def evaluate(self, genome: Any, rng: chex.PRNGKey | None = None) -> chex.Array:
        """Single genome evaluation fallback.

        Normally evaluate_population is used because we vmap over it.
        """
        # We don't implement the inner loop here because Lamarckian updates
        # are managed in evaluate_population where we have access to the Population object.
        raise NotImplementedError("LSMFEvaluator should be called via evaluate_population.")

    def evaluate_population(
        self, population: BasePopulation[Any], key: chex.PRNGKey | None = None
    ) -> BasePopulation[Any]:
        """Evaluates the population by first running SGD on weights and biases.

        Uses jax.vmap to run the optimization loop independently for every
        genome in the batch. Returns a Lamarckian-updated population.
        """

        if self.epochs <= 0:
            return self.base_evaluator.evaluate_population(population, key)

        # 1. Define the objective function w.r.t continuous parameters
        # Closing over the discrete topology (genome structure)
        def loss_fn(weights: chex.Array, biases: chex.Array, genome: Any, rng: chex.PRNGKey | None):
            temp_genome = genome.replace(weights=weights, biases=biases)
            if rng is not None:
                return self.base_evaluator.evaluate(temp_genome, rng)
            else:
                return self.base_evaluator.evaluate(temp_genome)

        # 3. Define a single optimization step
        def opt_step(state, _):
            weights, biases, opt_state, genome, rng = state

            # Compute gradients
            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(weights, biases, genome, rng)

            # Apply optax updates
            updates, new_opt_state = self.optimizer.update(grads, opt_state, (weights, biases))
            new_weights = optax.apply_updates(weights, updates[0])
            new_biases = optax.apply_updates(biases, updates[1])

            new_state = (new_weights, new_biases, new_opt_state, genome, rng)
            return new_state, loss

        # 4. Define the training loop for a single genome
        def train_individual(genome: Any, rng: chex.PRNGKey | None):
            opt_state = self.optimizer.init((genome.weights, genome.biases))
            initial_state = (genome.weights, genome.biases, opt_state, genome, rng)

            # Run self.epochs of SGD
            final_state, losses = jax.lax.scan(opt_step, initial_state, None, length=self.epochs)

            final_weights, final_biases, _, _, _ = final_state

            # The final fitness is the loss at the last epoch
            final_fitness = losses[-1]
            return final_weights, final_biases, final_fitness

        # 5. Map the training loop across all individuals in the population
        if key is not None:
            rngs = jax.random.split(key, population.fitness.shape[0])
            new_weights, new_biases, new_fitness = jax.vmap(train_individual)(population.genes, rngs)
        else:
            # We map without rng
            # Vmap needs identical structure, so we pass a dummy None to vmap via in_axes
            def train_wrapper(g):
                return train_individual(g, None)
            new_weights, new_biases, new_fitness = jax.vmap(train_wrapper)(population.genes)

        # 6. Lamarckian Update
        # Reconstruct the genome with the optimized weights
        new_genes = population.genes.replace(weights=new_weights, biases=new_biases)

        # Return the fully updated population
        return population.replace(genes=new_genes, fitness=new_fitness)

__all__ = ["LSMFEvaluator"]
