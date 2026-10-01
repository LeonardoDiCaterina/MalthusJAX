"""LSMF (Learn, Select, Mutate, Forget) Memetic Evaluator Wrapper.

Implements the "Learn" step from Märtens & Izzo (2019) by applying stochastic
gradient descent or Adam to continuous weights and biases within the JAX
compilation boundary, returning Lamarckian-updated individuals.
"""

from __future__ import annotations

from typing import Any, Tuple, cast

import chex
import jax
import optax
from flax import struct

from malthus_gp.genomes.differentiable_cartesian_genome import (
    DifferentiableCartesianGenome,
)
from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.evaluators.lsmf_evaluator")

_field: Any = struct.field


@struct.dataclass
class LSMFEvaluatorConfig(BaseEvaluatorConfig):  # type: ignore[misc]
    """Configuration for LSMF Evaluator."""

    epochs: int = _field(pytree_node=False, default=10)
    maximize: bool = _field(pytree_node=False, default=False)


@struct.dataclass
class LSMFEvaluator(
    BaseEvaluator[DifferentiableCartesianGenome, LSMFEvaluatorConfig, Any]  # type: ignore[misc]
):
    """Memetic evaluator applying local gradient descent (Adam/SGD) to continuous parameters.

    Attributes:
        base_evaluator: Target evaluator (supervised loss or ODE residual).
        optimizer: Optax gradient transformation (e.g. optax.adam(1e-2)).
    """

    base_evaluator: BaseEvaluator = _field(pytree_node=False)
    optimizer: optax.GradientTransformation = _field(
        pytree_node=False, default_factory=lambda: optax.adam(1e-2)
    )

    def evaluate(self, genome: DifferentiableCartesianGenome) -> chex.Numeric:
        """Evaluates a single genome by running gradient descent on weights/biases."""
        if self.config.epochs <= 0:
            return self.base_evaluator.evaluate(genome)

        def loss_fn(w: chex.Array, b: chex.Array) -> chex.Array:
            temp = genome.replace(weights=w, biases=b)
            return cast(chex.Array, self.base_evaluator.evaluate(temp))

        def opt_step(
            state: Tuple[chex.Array, chex.Array, Any], _: Any
        ) -> Tuple[Tuple[chex.Array, chex.Array, Any], chex.Array]:
            w, b, opt_state = state
            loss, (g_w, g_b) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b)
            updates, new_opt_state = self.optimizer.update((g_w, g_b), opt_state, (w, b))
            new_w = optax.apply_updates(w, updates[0])
            new_b = optax.apply_updates(b, updates[1])
            return (new_w, new_b, new_opt_state), loss

        opt_init = self.optimizer.init((genome.weights, genome.biases))
        initial_state = (genome.weights, genome.biases, opt_init)

        final_state, losses = jax.lax.scan(opt_step, initial_state, None, length=self.config.epochs)
        return losses[-1]

    def evaluate_population(
        self,
        population: BasePopulation[DifferentiableCartesianGenome],
        key: chex.PRNGKey | None = None,
    ) -> BasePopulation[DifferentiableCartesianGenome]:
        """Runs vectorized gradient descent across the population and returns Lamarckian updates."""
        if self.config.epochs <= 0:
            return self.base_evaluator.evaluate_population(population)

        def train_individual(
            genome: DifferentiableCartesianGenome,
        ) -> Tuple[chex.Array, chex.Array, chex.Array]:
            def loss_fn(w: chex.Array, b: chex.Array) -> chex.Array:
                temp = genome.replace(weights=w, biases=b)
                return cast(chex.Array, self.base_evaluator.evaluate(temp))

            def opt_step(
                state: Tuple[chex.Array, chex.Array, Any], _: Any
            ) -> Tuple[Tuple[chex.Array, chex.Array, Any], chex.Array]:
                w, b, opt_state = state
                loss, (g_w, g_b) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b)
                updates, new_opt_state = self.optimizer.update((g_w, g_b), opt_state, (w, b))
                new_w = optax.apply_updates(w, updates[0])
                new_b = optax.apply_updates(b, updates[1])
                return (new_w, new_b, new_opt_state), loss

            opt_init = self.optimizer.init((genome.weights, genome.biases))
            initial_state = (genome.weights, genome.biases, opt_init)

            final_state, losses = jax.lax.scan(
                opt_step, initial_state, None, length=self.config.epochs
            )
            final_w, final_b, _ = final_state
            return final_w, final_b, losses[-1]

        # Vectorize training loop across all individuals in the batch
        new_weights, new_biases, new_fitness = jax.vmap(train_individual)(population.genes)

        # Lamarckian update: update genomes with locally trained continuous weights
        new_genes = cast(
            DifferentiableCartesianGenome,
            population.genes.replace(weights=new_weights, biases=new_biases),
        )
        return cast(
            BasePopulation[DifferentiableCartesianGenome],
            cast(Any, population).replace(genes=new_genes, fitness=new_fitness),
        )
