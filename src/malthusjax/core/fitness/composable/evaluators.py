"""Composable Evaluator implementations.

Provides:
- ``SupervisedEvaluator``: runs SL evaluation loop (predict → loss).
- ``OptimizationEvaluator``: runs direct objective evaluation loop.

These evaluators follow the three-axis decomposition:
  Evaluator = Problem(TaskType × Environment × Interpreter) × Output

See docs/evaluator_design.md for the full design.
"""

from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.composable.base import (
    BaseInterpreter,
    BaseSupervisedEnvironment,
    BaseOptimizationEnvironment,
    ScalarOutput,
    QDOutput,
)


# =============================================================================
# SupervisedEvaluator
# =============================================================================

@struct.dataclass
class SupervisedEvaluator:
    """Composable evaluator for SupervisedTask problems.

    Orchestrates the supervised learning evaluation loop:
    1. Extract ``(X, y)`` from the environment.
    2. For each genome, apply the interpreter over X to get predictions.
    3. Compute the loss (via ``output.compute_loss``) and return fitness.

    Compatible with any combination of:
    - ``BaseSupervisedEnvironment`` (``SklearnEnv``, ``CustomDatasetEnv``, etc.)
    - ``BaseInterpreter`` (``MLPInterpreter``, ``LinearGPInterpreter``,
      ``TensorNEATInterpreter``, etc.)
    - ``ScalarOutput`` (or ``QDOutput``, ``MOOutput`` in future)

    Usage::

        env = SklearnEnv.create(dataset="breast_cancer")
        interp = MLPInterpreter(input_dim=30, output_dim=1, hidden=(64, 64))
        output = ScalarOutput(loss_fn="bce")
        evaluator = SupervisedEvaluator(env=env, interpreter=interp, output=output)

        population = evaluator.evaluate_population(population)
    """

    env: BaseSupervisedEnvironment = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    interpreter: BaseInterpreter = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    output: ScalarOutput = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: Any) -> chex.Numeric:
        """Evaluate a single genome on the supervised task.

        Applies the interpreter over the full dataset X (vmapped over rows),
        then computes the configured loss against y.

        Args:
            genome: A single (unbatched) genome compatible with ``self.interpreter``.

        Returns:
            Scalar fitness value.
        """
        X = self.env.X  # (n_samples, n_features)
        y = self.env.y  # (n_samples,)

        # Apply interpreter to each row of X
        # interpreter.apply(genome, x_i) → prediction_i
        predictions = jax.vmap(
            lambda x_i: self.interpreter.apply(genome, x_i)
        )(X)  # (n_samples, output_dim)

        return self.output.compute_loss(predictions, y)

    def evaluate_population(self, population: BasePopulation) -> BasePopulation:
        """Vectorized population evaluation via jax.vmap.

        Args:
            population: Population with ``.genes`` containing a batch of genomes.

        Returns:
            Population with updated ``.fitness`` array.
        """
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        return cast(Any, population).replace(fitness=fitness_scores)


# =============================================================================
# OptimizationEvaluator
# =============================================================================

@struct.dataclass
class OptimizationEvaluator:
    """Composable evaluator for OptimizationTask problems.

    Orchestrates the direct objective evaluation loop:
    1. Extract the raw solution vector from the genome (via IdentityInterpreter).
    2. Pass it directly to ``env.evaluate(solution)`` which applies f(x, data).
    3. Return the scalar objective as fitness.

    Suitable for both analytic black-box functions (BBOB, Sphere) and
    data-defined problem instances (TSP, Knapsack, Graph Coloring) — any
    environment where the genome encodes a candidate solution directly.

    Usage::

        env = BBOBEnv.create(fn_name="sphere", num_dims=10)
        interp = IdentityInterpreter()
        output = ScalarOutput()
        evaluator = OptimizationEvaluator(env=env, interpreter=interp, output=output)

        population = evaluator.evaluate_population(population)
    """

    env: BaseOptimizationEnvironment = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    interpreter: BaseInterpreter = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    output: ScalarOutput = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: Any) -> chex.Numeric:
        """Evaluate a single genome against the optimization problem instance.

        The interpreter is applied first (typically IdentityInterpreter which
        simply returns genome.values), then the solution is passed to
        env.evaluate().

        Args:
            genome: A single (unbatched) genome.

        Returns:
            Scalar objective value.
        """
        # For OptimizationTask, inputs=None — the genome IS the solution
        solution = self.interpreter.apply(genome, inputs=None)
        raw_score = self.env.evaluate(solution)
        return self.output.format_fitness(raw_score)

    def evaluate_population(self, population: BasePopulation) -> BasePopulation:
        """Vectorized population evaluation via jax.vmap.

        Args:
            population: Population with ``.genes`` containing a batch of genomes.

        Returns:
            Population with updated ``.fitness`` array.
        """
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        return cast(Any, population).replace(fitness=fitness_scores)
